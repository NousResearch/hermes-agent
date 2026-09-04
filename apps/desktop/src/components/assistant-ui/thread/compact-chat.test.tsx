// Compact chat hides thinking, tool, timer, and background-process rows at
// render time, and must still show approvals and agent-to-agent chips.
import { type ThreadMessage } from '@assistant-ui/react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $compactChat, setCompactChat } from '@/store/compact-chat'
import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'
import { clearDismissedToolRows } from '@/store/tool-dismiss'
import { $toolDisclosureStates } from '@/store/tool-view'

import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'

import { Thread } from '.'

const createdAt = new Date('2026-06-03T00:00:00.000Z')
const sessionId = 'sess-compact-chat'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

stubThreadEnvironment()
stubThreadViewportSize()
vi.stubGlobal('ResizeObserver', TestResizeObserver)

function user(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantWithWork(): ThreadMessage {
  return {
    id: 'assistant-work',
    role: 'assistant',
    content: [
      { type: 'reasoning', text: 'I should inspect the file first.' },
      {
        type: 'tool-call',
        toolCallId: 'read-1',
        toolName: 'read_file',
        args: { path: '/etc/hosts' },
        argsText: JSON.stringify({ path: '/etc/hosts' }),
        result: { content: '127.0.0.1 localhost' }
      },
      {
        type: 'tool-call',
        toolCallId: 'term-1',
        toolName: 'terminal',
        args: { command: 'ls -la' },
        argsText: JSON.stringify({ command: 'ls -la' }),
        result: { output: 'ok', exit_code: 0 }
      },
      { type: 'text', text: 'Hosts file looks fine.' }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: { durationS: 12 }
    }
  } as unknown as ThreadMessage
}

function pendingTerminal(): ThreadMessage {
  return {
    id: 'assistant-pending',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'term-pending',
        toolName: 'terminal',
        args: { command: 'rm -rf /tmp/x' },
        argsText: JSON.stringify({ command: 'rm -rf /tmp/x' })
      }
    ],
    status: { type: 'running' },
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

const Harness = ({ messages }: { messages: ThreadMessage[] }) => (
  <ThreadRuntime messages={messages}>
    <Thread />
  </ThreadRuntime>
)

beforeEach(() => {
  localStorage.clear()
  $compactChat.set(false)
  clearAllPrompts()
  $activeSessionId.set(sessionId)
  $toolDisclosureStates.set({})
  clearDismissedToolRows()
})

afterEach(() => {
  cleanup()
  $compactChat.set(false)
  clearAllPrompts()
  $activeSessionId.set(null)
  clearDismissedToolRows()
})

describe('compact chat (default off)', () => {
  it('renders thinking, tool chrome, and the turn-duration chip', async () => {
    const { container } = render(<Harness messages={[user('u1', 'check hosts'), assistantWithWork()]} />)

    expect(await screen.findByText('Hosts file looks fine.')).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_thinking-disclosure"]')).toBeTruthy()
    expect(container.querySelector('[data-tool-summary],[data-tool-row]')).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_turn-duration"]')).toBeTruthy()
  })
})

describe('compact chat (enabled)', () => {
  beforeEach(() => {
    setCompactChat(true)
  })

  it('hides thinking, tool chrome, and the turn-duration chip while keeping the answer', async () => {
    const { container } = render(<Harness messages={[user('u1', 'check hosts'), assistantWithWork()]} />)

    expect(await screen.findByText('Hosts file looks fine.')).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_thinking-disclosure"]')).toBeNull()
    expect(container.querySelector('[data-slot="aui_reasoning-text"]')).toBeNull()
    expect(container.querySelector('[data-tool-summary]')).toBeNull()
    expect(container.querySelector('[data-tool-row]')).toBeNull()
    expect(container.querySelector('[data-slot="aui_turn-duration"]')).toBeNull()
  })

  it('still renders an inline approval prompt', async () => {
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous command', sessionId })

    const { container } = render(<Harness messages={[user('u1', 'clean tmp'), pendingTerminal()]} />)

    await waitFor(() => {
      expect(container.querySelector('[data-slot="tool-approval-inline"]')).not.toBeNull()
    })
  })

  it('still renders agent-to-agent message chips', async () => {
    const { container } = render(
      <Harness messages={[user('u1', 'Message from 🤖 Hermes: ping'), assistantWithWork()]} />
    )

    expect(await screen.findByText('Hosts file looks fine.')).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_agent-message-note"]')).toBeTruthy()
    expect(container.textContent).toContain('Message from Hermes')
  })

  it('hides background-process notice rows', async () => {
    const { container } = render(
      <Harness
        messages={[
          user('u1', '[IMPORTANT: Background process proc_1 exited with code 0]'),
          assistantWithWork()
        ]}
      />
    )

    expect(await screen.findByText('Hosts file looks fine.')).toBeTruthy()
    expect(container.textContent).not.toContain('Background process proc_1')
  })
})
