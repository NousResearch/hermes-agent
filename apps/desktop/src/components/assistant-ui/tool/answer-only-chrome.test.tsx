import { type ThreadMessage } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '@/components/assistant-ui/test-utils'
import { Thread } from '@/components/assistant-ui/thread'
import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $showReasoning } from '@/store/reasoning-disclosure'
import { $activeSessionId } from '@/store/session'

stubThreadEnvironment()
stubThreadViewportSize()

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

beforeEach(() => {
  clearAllPrompts()
  $activeSessionId.set('sess-1')
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
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous command', sessionId: 'sess-1' })

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
})
