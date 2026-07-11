import { act, cleanup, render } from '@testing-library/react'
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { ResponseLoadingIndicator } from './status'

// Integration coverage for the elapsed-timer navigation fix (hermes-agent#62158):
// `ResponseLoadingIndicator` must resolve a stable `run:${messageId}` key from
// the running assistant message so the timer survives a view navigation
// (remount) instead of restarting from zero, and must restart when a new run
// (new assistant message id) begins. The component under test is the real
// `ResponseLoadingIndicator`; keying flows through the shared keyed
// `useElapsedSeconds` in activity-timer.ts.
function Harness({ messageId }: { messageId: string }) {
  const createdAt = new Date('2026-01-01T00:00:00.000Z')
  const messages: ThreadMessage[] = [
    {
      id: 'user-1',
      role: 'user',
      content: [{ type: 'text', text: 'hi' }],
      createdAt,
      metadata: {
        unstable_state: null,
        unstable_annotations: [],
        unstable_data: [],
        steps: [],
        custom: {}
      },
      status: { type: 'complete', reason: 'stop' }
    },
    {
      id: messageId,
      role: 'assistant',
      content: [{ type: 'text', text: '' }],
      createdAt,
      metadata: {
        unstable_state: 'running',
        unstable_annotations: [],
        unstable_data: [],
        steps: [],
        custom: {}
      },
      status: { type: 'running' }
    }
  ] as ThreadMessage[]

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: true,
    onNew: async () => {}
  })

  return (
    <I18nProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <ResponseLoadingIndicator />
      </AssistantRuntimeProvider>
    </I18nProvider>
  )
}

// Simulates the transition right after a new prompt is sent but before the
// runtime appends the run's assistant message: the last message is a user
// prompt (or a prior turn's completed assistant), with no in-flight assistant
// message yet. The indicator must NOT key on a stale prior-turn id.
function TransitionHarness() {
  const createdAt = new Date('2026-01-01T00:00:00.000Z')
  const messages: ThreadMessage[] = [
    {
      id: 'user-1',
      role: 'user',
      content: [{ type: 'text', text: 'hi' }],
      createdAt,
      metadata: {
        unstable_state: null,
        unstable_annotations: [],
        unstable_data: [],
        steps: [],
        custom: {}
      },
      status: { type: 'complete', reason: 'stop' }
    },
    {
      id: 'assistant-prior',
      role: 'assistant',
      content: [{ type: 'text', text: 'done' }],
      createdAt,
      metadata: {
        unstable_state: null,
        unstable_annotations: [],
        unstable_data: [],
        steps: [],
        custom: {}
      },
      status: { type: 'complete', reason: 'stop' }
    },
    {
      id: 'user-2',
      role: 'user',
      content: [{ type: 'text', text: 'new prompt' }],
      createdAt,
      metadata: {
        unstable_state: null,
        unstable_annotations: [],
        unstable_data: [],
        steps: [],
        custom: {}
      },
      status: { type: 'complete', reason: 'stop' }
    }
  ] as ThreadMessage[]

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: true,
    onNew: async () => {}
  })

  return (
    <I18nProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <ResponseLoadingIndicator />
      </AssistantRuntimeProvider>
    </I18nProvider>
  )
}

// Last message is a PRIOR turn's completed assistant (no running message
// appended yet) — the other transition shape Flash flagged. Must show a fresh
// timer, not a stale timestamp from the completed turn.
function CompletedLastHarness() {
  const createdAt = new Date('2026-01-01T00:00:00.000Z')
  const messages: ThreadMessage[] = [
    {
      id: 'user-1',
      role: 'user',
      content: [{ type: 'text', text: 'hi' }],
      createdAt,
      metadata: {
        unstable_state: null,
        unstable_annotations: [],
        unstable_data: [],
        steps: [],
        custom: {}
      },
      status: { type: 'complete', reason: 'stop' }
    },
    {
      id: 'assistant-prior',
      role: 'assistant',
      content: [{ type: 'text', text: 'done' }],
      createdAt,
      metadata: {
        unstable_state: null,
        unstable_annotations: [],
        unstable_data: [],
        steps: [],
        custom: {}
      },
      status: { type: 'complete', reason: 'stop' }
    }
  ] as ThreadMessage[]

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: true,
    onNew: async () => {}
  })

  return (
    <I18nProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <ResponseLoadingIndicator />
      </AssistantRuntimeProvider>
    </I18nProvider>
  )
}

describe('ResponseLoadingIndicator elapsed timer', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'))
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('keeps counting across a remount while the same run is active', () => {
    const first = render(<Harness messageId="run-a" />)

    act(() => {
      vi.advanceTimersByTime(9_000)
    })

    expect(first.container.textContent).toMatch(/9s/)

    first.unmount()

    act(() => {
      vi.advanceTimersByTime(4_000)
    })

    const second = render(<Harness messageId="run-a" />)
    expect(second.container.textContent).toMatch(/13s/)
  })

  it('resets when a new run (new assistant message id) starts', () => {
    const first = render(<Harness messageId="run-a" />)

    act(() => {
      vi.advanceTimersByTime(10_000)
    })

    expect(first.container.textContent).toMatch(/10s/)

    cleanup()

    const second = render(<Harness messageId="run-b" />)
    expect(second.container.textContent).toMatch(/0s/)
  })

  it('does not reuse a stale prior-turn timestamp during the pre-append transition', () => {
    // First, let a prior turn accumulate elapsed time under its own key.
    const prior = render(<Harness messageId="assistant-prior" />)
    act(() => {
      vi.advanceTimersByTime(30_000)
    })
    expect(prior.container.textContent).toMatch(/30s/)
    prior.unmount()

    // New prompt sent, but the runtime has not appended the run's assistant
    // message yet: the last message is the user prompt. The indicator must
    // start a FRESH timer (0s), not reuse the prior turn's 30s timestamp.
    const transition = render(<TransitionHarness />)
    expect(transition.container.textContent).toMatch(/0s/)
  })

  it('does not reuse a stale timestamp when the last message is a completed prior-turn assistant', () => {
    // Let a prior turn accumulate elapsed time under its own key.
    const prior = render(<Harness messageId="assistant-prior" />)
    act(() => {
      vi.advanceTimersByTime(45_000)
    })
    expect(prior.container.textContent).toMatch(/45s/)
    prior.unmount()

    // The runtime has not appended the new run's assistant message yet, so the
    // last message is the previous completed assistant. Must start fresh (0s).
    const completed = render(<CompletedLastHarness />)
    expect(completed.container.textContent).toMatch(/0s/)
  })
})
