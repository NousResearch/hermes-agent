import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $now, $scheduledRetries, setScheduledRetry } from '@/store/scheduled-retry'
import { $activeSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { ScheduledRetryScheduler } from './scheduled-retry-scheduler'

stubThreadEnvironment()
stubThreadViewportSize()

const createdAt = new Date('2026-05-01T00:00:00.000Z')

const failedMessage = {
  id: 'assistant-error-1',
  role: 'assistant',
  content: [],
  status: { type: 'incomplete', reason: 'error', error: 'HTTP 429' },
  createdAt,
  metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} },
  parentId: 'user-1'
} as unknown as ThreadMessage

const userMessage = {
  id: 'user-1',
  role: 'user',
  content: [{ type: 'text', text: 'go' }],
  attachments: [],
  createdAt,
  metadata: { custom: {} }
} as ThreadMessage

function Harness({ onReload }: { onReload: () => void }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage, failedMessage],
    isRunning: false,
    onNew: async () => {},
    onReload: async () => onReload()
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ScheduledRetryScheduler />
    </AssistantRuntimeProvider>
  )
}

describe('ScheduledRetryScheduler', () => {
  beforeEach(() => {
    window.localStorage.removeItem('hermes.desktop.scheduledRetries')
    $scheduledRetries.set({})
    $activeSessionId.set('session-1')
    $now.set(Date.now())
    $sessionStates.set({})
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('fires the reload when the schedule is due and clears the record', async () => {
    const onReload = vi.fn()
    render(<Harness onReload={onReload} />)

    await act(async () => {
      setScheduledRetry('session-1', {
        at: Date.now() - 1000,
        messageId: 'assistant-error-1',
        sessionId: 'session-1'
      })
    })

    await act(async () => {
      $now.set(Date.now())
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })

    expect(onReload).toHaveBeenCalledTimes(1)
    expect($scheduledRetries.get()['session-1']).toBeUndefined()
  })

  it('retires a schedule whose message is gone', async () => {
    const onReload = vi.fn()

    function EmptyHarness() {
      const runtime = useExternalStoreRuntime<ThreadMessage>({
        messages: [userMessage],
        isRunning: false,
        onNew: async () => {},
        onReload: async () => onReload()
      })

      return (
        <AssistantRuntimeProvider runtime={runtime}>
          <ScheduledRetryScheduler />
        </AssistantRuntimeProvider>
      )
    }

    render(<EmptyHarness />)

    await act(async () => {
      setScheduledRetry('session-1', {
        at: Date.now() - 1000,
        messageId: 'assistant-error-1',
        sessionId: 'session-1'
      })
    })

    await act(async () => {
      $now.set(Date.now())
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })

    expect(onReload).not.toHaveBeenCalled()
    expect($scheduledRetries.get()['session-1']).toBeUndefined()
  })

  it('re-arms instead of firing into a busy session', async () => {
    const onReload = vi.fn()
    $sessionStates.set({ 'session-1': { busy: true } as never })

    render(<Harness onReload={onReload} />)

    await act(async () => {
      setScheduledRetry('session-1', {
        at: Date.now() - 1000,
        messageId: 'assistant-error-1',
        sessionId: 'session-1'
      })
    })

    await act(async () => {
      $now.set(Date.now())
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })

    expect(onReload).not.toHaveBeenCalled()
    // The schedule survived, re-armed into the future.
    expect($scheduledRetries.get()['session-1']?.at).toBeGreaterThan(Date.now())
  })

  it('does nothing without a pending schedule', async () => {
    const onReload = vi.fn()
    render(<Harness onReload={onReload} />)

    await act(async () => {
      $now.set(Date.now())
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })

    expect(onReload).not.toHaveBeenCalled()
  })
})
