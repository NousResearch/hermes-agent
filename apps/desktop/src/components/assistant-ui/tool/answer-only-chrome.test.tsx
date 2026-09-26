import { type ThreadMessage } from '@assistant-ui/react'
import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { renderMessageStream } from '@/app/session/hooks/use-message-stream/test-harness'
import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '@/components/assistant-ui/test-utils'
import { Thread } from '@/components/assistant-ui/thread'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $showReasoning } from '@/store/reasoning-disclosure'
import { $activeSessionId } from '@/store/session'

stubThreadEnvironment()
stubThreadViewportSize()

const SID = 'sess-1'
const createdAt = new Date('2026-06-03T00:00:00.000Z')

function answerOnlyMessage(): ThreadMessage {
  return {
    id: 'assistant-answer-only',
    role: 'assistant',
    content: [
      {
        type: 'reasoning',
        text: 'hidden chain of thought',
        timestamp: createdAt.getTime() / 1000 + 1,
        completedAt: createdAt.getTime() / 1000 + 2
      },
      {
        type: 'tool-call',
        toolCallId: 'read-1',
        toolName: 'read_file',
        args: { path: '/repo/src/status.tsx' },
        argsText: JSON.stringify({ path: '/repo/src/status.tsx' }),
        result: { content: 'export const Status = () => null' }
      },
      {
        type: 'tool-call',
        toolCallId: 'search-1',
        toolName: 'search_files',
        args: { query: 'toolRuns' },
        argsText: JSON.stringify({ query: 'toolRuns' }),
        result: { matches: [] }
      },
      {
        type: 'tool-call',
        toolCallId: 'fail-1',
        toolName: 'terminal',
        args: { command: 'deploy' },
        argsText: JSON.stringify({ command: 'deploy' }),
        isError: true,
        result: { error: 'disk full, act now' }
      },
      {
        type: 'text',
        text: 'final answer only'
      }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as unknown as ThreadMessage
}

function Harness() {
  return (
    <ThreadRuntime messages={[answerOnlyMessage()]}>
      <Thread />
    </ThreadRuntime>
  )
}

// Drive the gateway's own tool.complete event through the message stream (the
// store Desktop renders from), then render what it produced. Answer-only
// suppresses tool.start, so the completion arrives on its own, and its failure
// sits inside `result`: nothing hand-sets isError on the part.
function completionHarness(payload: Record<string, unknown>) {
  const stream = renderMessageStream(SID)

  const send = (type: GatewayEvent['type'], body: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ payload: body, session_id: SID, type }))

  send('message.start')
  send('tool.complete', payload)
  send('message.complete', { text: 'done' })

  return (
    <ThreadRuntime messages={stream.state(SID).messages.map(toRuntimeMessage)}>
      <Thread />
    </ThreadRuntime>
  )
}

beforeEach(() => {
  clearAllPrompts()
  $activeSessionId.set(SID)
  $showReasoning.set(true)
})

afterEach(() => {
  cleanup()
  clearAllPrompts()
  $activeSessionId.set(null)
  $showReasoning.set(true)
})

describe('answer-only display policy', () => {
  it('hides reasoning and non-essential tool chrome without requiring reasoning_effort none', async () => {
    $showReasoning.set(false)
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous command', sessionId: SID })

    const { container } = render(<Harness />)

    expect(await screen.findByText('final answer only')).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_thinking-disclosure"]')).toBeNull()
    expect(container.querySelector('[data-tool-summary]')).toBeNull()
    expect(screen.getByRole('button', { name: /Run/ })).toBeTruthy()
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(1)
  })

  it('still shows tool chrome when reasoning blocks are on', async () => {
    const { container } = render(<Harness />)

    expect(await screen.findByText(/Explored 2 files/)).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_thinking-disclosure"]')).not.toBeNull()
  })

  it('keeps a failed call whose error sits inside result, from a real tool.complete payload', async () => {
    // The gateway's tool.complete never sets a top-level error: a read_file
    // failure rides inside result. The answer-only gate must still show it.
    $showReasoning.set(false)

    const { container } = render(
      completionHarness({
        name: 'read_file',
        tool_id: 'read-fail-1',
        args: { path: '/repo/src/status.tsx' },
        result: { error: 'disk full, act now' }
      })
    )

    expect(await screen.findByText('done')).toBeTruthy()
    const rows = container.querySelectorAll('[data-tool-row]')
    expect(rows).toHaveLength(1)
  })

  it('keeps a failed terminal call with a non-zero exit_code, from a real tool.complete payload', async () => {
    $showReasoning.set(false)

    const { container } = render(
      completionHarness({
        name: 'terminal',
        tool_id: 'term-fail-1',
        args: { command: 'deploy' },
        result: { output: 'Error: deploy failed', exit_code: 1, error: null }
      })
    )

    expect(await screen.findByText('done')).toBeTruthy()
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(1)
  })

  it('keeps a call that reports success: false, from a real tool.complete payload', async () => {
    $showReasoning.set(false)

    const { container } = render(
      completionHarness({
        // Not a card tool: file edits stay visible anyway, so they can't prove the failure path.
        name: 'web_extract',
        tool_id: 'extract-fail-1',
        args: { urls: ['https://example.test'] },
        result: { success: false }
      })
    )

    expect(await screen.findByText('done')).toBeTruthy()
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(1)
  })

  it('still hides a successful call driven through the same tool.complete mapping', async () => {
    $showReasoning.set(false)

    const { container } = render(
      completionHarness({
        name: 'read_file',
        tool_id: 'read-ok-1',
        args: { path: '/repo/src/status.tsx' },
        result: { content: 'export const Status = () => null' }
      })
    )

    expect(await screen.findByText('done')).toBeTruthy()
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(0)
  })
})
