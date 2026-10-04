import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $trajectoryCollapsedByDefault } from '@/store/trajectory-disclosure'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'
import { planTrajectoryCollapse } from './trajectory-collapse'

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const options = { messageComplete: true, preferenceOn: true }

stubThreadEnvironment()

afterEach(cleanup)
beforeEach(() => $trajectoryCollapsedByDefault.set(true))

function assistantMessage(running = false): ThreadMessage {
  const start = createdAt.getTime() / 1000
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [
      { completedAt: start + 1, text: 'plan', timestamp: start, type: 'reasoning' },
      {
        args: { path: '/etc/hosts' },
        argsText: JSON.stringify({ path: '/etc/hosts' }),
        completedAt: start + 2,
        result: { content: '127.0.0.1 localhost' },
        timestamp: start + 1,
        toolCallId: 'read-1',
        toolName: 'read_file',
        type: 'tool-call'
      },
      { text: 'final answer here', type: 'text' }
    ],
    createdAt,
    metadata: { custom: {}, steps: [], unstable_annotations: [], unstable_data: [], unstable_state: null },
    status: running ? { type: 'running' } : { reason: 'stop', type: 'complete' }
  } as ThreadMessage
}

function Harness({ assistant }: { assistant: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: assistant.status?.type === 'running',
    messages: [
      {
        attachments: [],
        content: [{ text: 'what should I do?', type: 'text' }],
        createdAt,
        id: 'user-1',
        role: 'user'
      },
      assistant
    ] as ThreadMessage[],
    onNew: async () => {}
  })
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('execution trajectory collapse', () => {
  it('folds completed preliminary groups without unmounting them', async () => {
    const { container } = render(<Harness assistant={assistantMessage()} />)

    const summary = await screen.findByRole('button', { name: /Completed 2 steps in 2s/i })
    expect(screen.getByText('final answer here')).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_thinking-disclosure"]')).toBeTruthy()
    expect(container.querySelector('[data-tool-row]')).toBeTruthy()
    expect(container.querySelector('[data-trajectory-group="reasoning"]')?.hasAttribute('hidden')).toBe(true)
    expect(container.querySelector('[data-trajectory-group="tool"]')?.hasAttribute('hidden')).toBe(true)

    fireEvent.click(summary)

    expect(container.querySelector('[data-slot="aui_thinking-disclosure"]')).toBeTruthy()
    expect(container.querySelector('[data-tool-row]')).toBeTruthy()
    expect(container.querySelector('[data-trajectory-group="reasoning"]')?.hasAttribute('hidden')).toBe(false)
    expect(container.querySelector('[data-trajectory-group="tool"]')?.hasAttribute('hidden')).toBe(false)
  })

  it('leaves a running turn and a disabled preference expanded', async () => {
    const { container, rerender } = render(<Harness assistant={assistantMessage(true)} />)

    await screen.findByText('final answer here')
    expect(screen.queryByRole('button', { name: /Completed .* steps/i })).toBeNull()
    expect(container.querySelector('[data-tool-row]')).toBeTruthy()

    $trajectoryCollapsedByDefault.set(false)
    rerender(<Harness assistant={assistantMessage()} />)
    expect(screen.queryByRole('button', { name: /Completed .* steps/i })).toBeNull()
    expect(container.querySelector('[data-tool-row]')).toBeTruthy()
  })

  it('uses a duration fallback and keeps successful image deliverables visible', () => {
    expect(
      planTrajectoryCollapse(
        [
          { text: 'plan', type: 'reasoning' },
          { toolName: 'read_file', type: 'tool-call' },
          { text: 'done', type: 'text' }
        ],
        { ...options, fallbackElapsedSeconds: 7.6 }
      )
    ).toEqual({ elapsedSeconds: 8, stepCount: 2 })
    expect(
      planTrajectoryCollapse(
        [
          { text: 'draw', type: 'reasoning' },
          {
            result: { image: 'https://example.com/image.png', success: true },
            toolName: 'image_generate',
            type: 'tool-call'
          },
          { text: 'done', type: 'text' }
        ],
        options
      )
    ).toBeNull()
  })
})
