import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Thread } from '.'

const createdAt = new Date('2026-07-20T12:00:00.000Z')

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
vi.stubGlobal('CSS', { escape: (str: string) => str })

Element.prototype.scrollTo = function scrollTo() {}

afterEach(cleanup)

function userMessage(text: string): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function Harness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: false,
    onNew: async () => {},
    onEdit: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

const SINGLE_PAYLOAD = `[ASYNC DELEGATION COMPLETE — deleg_abc]
A background subagent you dispatched earlier has finished.

Dispatched: 2026-07-20 11:59:42 (18s ago)
Original goal: Audit the session lifecycle
Context you provided: desktop transcript
Toolsets: terminal
Role: reviewer   Model: gpt-test
Status: completed   API calls: 4   Duration: 18.4s
--- RESULT ---
Found one stale-session edge case and added a regression test.
Second result line remains available in the full payload.`

const BATCH_PAYLOAD = `[ASYNC DELEGATION BATCH COMPLETE — deleg_batch]
A background fan-out of 2 subagent(s) you dispatched earlier has finished.

Context you provided: compare both render paths
Role: leaf   Model: gpt-test   Total duration: 23.1s

--- ✓ TASK 1/2: Inspect the single result  (status=completed, api_calls=2, 12.2s) ---
Single path is sound.

--- ✗ TASK 2/2: Inspect the batch result  (status=failed, api_calls=1, 9.8s) ---
(failed: fixture failure)
Partial output:
Batch parser reached the fallback.`

describe('async delegation completion transcript card', () => {
  it('renders a compact single completion and reveals the exact raw payload on demand', () => {
    const message = userMessage(SINGLE_PAYLOAD)
    render(<Harness message={message} />)

    expect(screen.getByText('Subagent completed')).toBeTruthy()
    expect(screen.getByText('completed')).toBeTruthy()
    expect(screen.getByText('Audit the session lifecycle')).toBeTruthy()
    expect(screen.getByText('18.4s')).toBeTruthy()
    expect(screen.getByText('Found one stale-session edge case and added a regression test.')).toBeTruthy()

    const toggle = screen.getByRole('button', { name: 'Show full payload' })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText(SINGLE_PAYLOAD)).toBeNull()

    fireEvent.click(toggle)

    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(toggle.ownerDocument.getElementById(toggle.getAttribute('aria-controls')!)?.textContent).toBe(SINGLE_PAYLOAD)
    expect(screen.getByRole('button', { name: 'Hide full payload' })).toBeTruthy()

    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText(SINGLE_PAYLOAD)).toBeNull()

    expect(message.role).toBe('user')
    expect(message.content).toEqual([{ type: 'text', text: SINGLE_PAYLOAD }])
  })

  it('renders a compact batch completion with aggregate state and task context', () => {
    render(<Harness message={userMessage(BATCH_PAYLOAD)} />)

    expect(screen.getByText('Subagents completed')).toBeTruthy()
    expect(screen.getByText('1 completed · 1 failed')).toBeTruthy()
    expect(screen.getByText('Inspect the single result')).toBeTruthy()
    expect(screen.getByText('23.1s')).toBeTruthy()
    expect(screen.getByText('Single path is sound.')).toBeTruthy()
    expect(screen.queryByText(BATCH_PAYLOAD)).toBeNull()
  })

  it('keeps a marker-matched malformed payload usable and expandable', () => {
    const malformed = '[ASYNC DELEGATION COMPLETE — deleg_broken]\nnot structured, but still important'
    render(<Harness message={userMessage(malformed)} />)

    expect(screen.getByText('Subagent completed')).toBeTruthy()
    expect(screen.getByText('not structured, but still important')).toBeTruthy()

    const toggle = screen.getByRole('button', { name: 'Show full payload' })
    fireEvent.click(toggle)
    expect(toggle.ownerDocument.getElementById(toggle.getAttribute('aria-controls')!)?.textContent).toBe(malformed)
  })

  it('does not reclassify ordinary user text that merely contains the marker phrase', () => {
    const ordinary = 'Please explain [ASYNC DELEGATION COMPLETE — deleg_abc] without treating this as an event.'
    render(<Harness message={userMessage(ordinary)} />)

    expect(screen.queryByText('Subagent completed')).toBeNull()
    expect(screen.getByRole('button', { name: 'Edit message' })).toBeTruthy()
    expect(screen.getByText(ordinary)).toBeTruthy()
  })
})
