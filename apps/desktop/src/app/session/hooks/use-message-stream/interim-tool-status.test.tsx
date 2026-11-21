import { AssistantRuntimeProvider, useAuiState, useExternalStoreRuntime } from '@assistant-ui/react'
import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize } from '@/components/assistant-ui/test-utils'
import { Thread } from '@/components/assistant-ui/thread'
import { buildToolView } from '@/components/assistant-ui/tool/fallback-model'
import type { ChatMessage } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'

import { finalizeStoppedMessages } from '../use-prompt-actions/rewind'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

stubThreadEnvironment()
stubThreadViewportSize()

function RuntimeMarker() {
  const running = useAuiState(state => state.thread.isRunning)

  return <span data-testid="running-marker">{String(running)}</span>
}

function Transcript({ messages, isRunning }: { messages: ChatMessage[]; isRunning: boolean }) {
  const runtime = useExternalStoreRuntime({
    messages: messages.map(toRuntimeMessage),
    isRunning,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
      <RuntimeMarker />
    </AssistantRuntimeProvider>
  )
}

const SID = 'interim-tool-status'
let stream: MessageStreamHarness

const event = (type: GatewayEventName, timestamp: number, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ payload: { ...payload, timestamp }, session_id: SID, type }))

const toolRows = () =>
  stream.state(SID).messages.flatMap(message => message.parts.filter(part => part.type === 'tool-call'))

const startTool = () => {
  event('message.start', 100)
  event('message.delta', 101, { text: 'Checking the output.' })
  event('tool.start', 102, { args: { command: 'echo done' }, name: 'terminal', tool_id: 'pending-tool' })
}

describe('interim commentary does not finish a running tool (#134342)', () => {
  beforeEach(() => {
    stream = renderMessageStream(SID)
  })
  afterEach(cleanup)

  it('keeps the tool running through duplicate interims until its result arrives', async () => {
    startTool()
    const transcript = render(<Transcript isRunning messages={stream.state(SID).messages} />)

    for (const timestamp of [103, 104]) {
      event('message.interim', timestamp, { text: 'Checking the output.' })
      const rows = toolRows()
      expect(rows).toHaveLength(1)
      expect(buildToolView(rows[0], '').status).toBe('running')
      expect(buildToolView(rows[0], '').title).not.toBe('Result unavailable')
      transcript.rerender(<Transcript isRunning messages={stream.state(SID).messages} />)
      const toolBlock = transcript.container.querySelector('[data-slot="tool-block"]') as HTMLElement
      expect(await within(toolBlock).findByRole('status', { name: 'Running' })).toBeTruthy()
      expect(screen.queryByText('Result unavailable')).toBeNull()
      expect(stream.state(SID).streamId).toBeNull()
      expect(stream.state(SID).messages[0].pending).toBe(false)
    }

    event('tool.complete', 110, { name: 'terminal', result: { output: 'done', exit_code: 0 }, tool_id: 'pending-tool' })
    const rows = toolRows()
    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({ completedAt: 110, result: { output: 'done', exit_code: 0 } })
    expect(buildToolView(rows[0], '').status).toBe('success')
    expect(stream.state(SID).messages).toHaveLength(1)
    transcript.rerender(<Transcript isRunning messages={stream.state(SID).messages} />)
    expect(screen.queryByText('Result unavailable')).toBeNull()
    const toolBlock = transcript.container.querySelector('[data-slot="tool-block"]') as HTMLElement
    await waitFor(() => expect(within(toolBlock).queryByRole('status', { name: 'Running' })).toBeNull())
  })

  it('still shows a missing result once the whole turn finishes', () => {
    startTool()
    event('message.interim', 103, { text: 'Checking the output.' })
    event('message.complete', 110, { text: 'Finished checking.' })

    const rows = toolRows()
    expect(rows).toHaveLength(1)
    expect(buildToolView(rows[0], '')).toMatchObject({ status: 'warning', title: 'Result unavailable' })
    expect(stream.state(SID).busy).toBe(false)
    render(<Transcript isRunning={false} messages={stream.state(SID).messages} />)
    expect(screen.getByText('Result unavailable')).toBeTruthy()
    expect(screen.queryByRole('status', { name: 'Running' })).toBeNull()
  })

  for (const ending of ['error', 'stop'] as const) {
    it(`does not restart a previous interim tool after ${ending} when a new turn begins`, async () => {
      startTool()
      event('message.interim', 103, { text: 'Checking the output.' })

      if (ending === 'error') {
        event('error', 110, { message: 'Synthetic terminal failure' })
      } else {
        act(() => {
          const state = stream.state(SID)
          stream.states.set(SID, {
            ...state,
            messages: finalizeStoppedMessages(state.messages, state.streamId, 110),
            busy: false,
            awaitingResponse: false,
            streamId: null,
            pendingBranchGroup: null,
            needsInput: false,
            interrupted: true,
            turnStartedAt: null,
            turnLive: false
          })
        })
      }

      expect(stream.state(SID).busy).toBe(false)
      const transcript = render(<Transcript isRunning={false} messages={stream.state(SID).messages} />)
      const settledTitle = ending === 'stop' ? 'Interrupted' : 'Result unavailable'
      expect(screen.getByText(settledTitle)).toBeTruthy()
      // A new submitted prompt makes this same thread live again. Historical messages persist.
      await act(async () => {
        transcript.rerender(<Transcript isRunning messages={stream.state(SID).messages} />)
      })
      await waitFor(() => expect(screen.getByTestId('running-marker').textContent).toBe('true'))
      const oldTool = transcript.container.querySelector('[data-slot="tool-block"]') as HTMLElement
      expect(within(oldTool).queryByRole('status', { name: 'Running' })).toBeNull()
      expect(screen.getByText(settledTitle)).toBeTruthy()
    })
  }
})
