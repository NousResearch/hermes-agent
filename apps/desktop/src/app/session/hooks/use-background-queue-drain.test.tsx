import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  MAX_AUTO_DRAIN_ATTEMPTS,
  ORPHANED_QUEUE_MAX_AGE_MS,
  parkQueuedPrompts
} from '@/store/composer-queue'
import { $notifications, clearNotifications } from '@/store/notifications'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { useBackgroundQueueDrain } from './use-background-queue-drain'
import type { SubmitTextOptions } from './use-prompt-actions/utils'

function Harness({
  enabled = true,
  runtimeMap,
  selectedStoredSessionId = 'stored-session-b',
  submitText
}: {
  enabled?: boolean
  runtimeMap: MutableRefObject<Map<string, string>>
  selectedStoredSessionId?: string | null
  submitText: (text: string, options?: SubmitTextOptions) => Promise<boolean> | boolean
}) {
  useBackgroundQueueDrain({
    enabled,
    runtimeIdByStoredSessionIdRef: runtimeMap,
    selectedStoredSessionId,
    submitText
  })

  return null
}

describe('useBackgroundQueueDrain', () => {
  beforeEach(() => {
    vi.useRealTimers()
    clearAllSessionStates()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.useRealTimers()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    clearNotifications()
    clearAllSessionStates()
  })

  // Drive the auto-drain retry ladder to exhaustion for an always-failing submit.
  async function exhaustDrainAttempts() {
    for (let attempt = 0; attempt < MAX_AUTO_DRAIN_ATTEMPTS; attempt++) {
      await act(async () => {
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(750)
      })
    }

    await act(async () => {
      await Promise.resolve()
    })
  }

  it('garbage-collects a stale undrainable entry after exhausting attempts, without toasting', async () => {
    vi.useFakeTimers()

    // No runtime mapping AND submit always rejects: the stored session can never
    // be resolved or resumed — a provably orphaned queue.
    const runtimeMap = { current: new Map<string, string>() }
    const submitText = vi.fn(async () => false)

    // Backdate the entry past the orphan cutoff so it is eligible for GC.
    $queuedPromptsBySession.set({
      'stored-session-a': [
        {
          attachments: [],
          id: 'q-orphan',
          queuedAt: Date.now() - ORPHANED_QUEUE_MAX_AGE_MS - 1000,
          text: 'orphaned prompt'
        }
      ]
    })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await exhaustDrainAttempts()

    expect(getQueuedPrompts('stored-session-a')).toHaveLength(0)
    expect($notifications.get()).toHaveLength(0)
  })

  it('keeps a recent undrainable entry queued and toasts instead of GC-ing it', async () => {
    vi.useFakeTimers()

    const runtimeMap = { current: new Map<string, string>() }
    const submitText = vi.fn(async () => false)

    // A fresh entry may just be failing transiently (backend briefly down); it
    // must survive and surface the "queue stuck" toast, not be discarded.
    $queuedPromptsBySession.set({
      'stored-session-a': [{ attachments: [], id: 'q-recent', queuedAt: Date.now(), text: 'recent prompt' }]
    })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await exhaustDrainAttempts()

    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
    expect($notifications.get().length).toBeGreaterThan(0)
  })

  it('drains an idle queued prompt for a non-selected background session', async () => {
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => true)

    enqueueQueuedPrompt('stored-session-a', { text: 'continue in the background', attachments: [] })
    clearAllSessionStates()

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await waitFor(() => {
      expect(submitText).toHaveBeenCalledWith('continue in the background', {
        attachments: [],
        fromQueue: true,
        sessionId: 'rt-session-a',
        storedSessionId: 'stored-session-a'
      })
    })

    await waitFor(() => expect(getQueuedPrompts('stored-session-a')).toHaveLength(0))
  })

  it('leaves the selected session queue to the mounted ChatBar drainer', async () => {
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => true)

    enqueueQueuedPrompt('stored-session-a', { text: 'visible queue entry', attachments: [] })
    clearAllSessionStates()

    render(<Harness runtimeMap={runtimeMap} selectedStoredSessionId="stored-session-a" submitText={submitText} />)

    await new Promise(resolve => window.setTimeout(resolve, 0))

    expect(submitText).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })

  it('does not drain a background session that is still marked working', async () => {
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => true)

    enqueueQueuedPrompt('stored-session-a', { text: 'wait for current turn', attachments: [] })
    // Mark the session as working (busy) so the drain should skip it
    publishSessionState('rt-session-a', { ...createClientSessionState('stored-session-a'), busy: true })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await new Promise(resolve => window.setTimeout(resolve, 0))

    expect(submitText).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })

  it('does not drain a parked background session, even when idle', async () => {
    // A Stop in a tile parks that session's queue; when the user then focuses
    // another chat, THIS drainer takes over the tile's queue — it must honor
    // the park just like the mounted ChatBar drainer does.
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => true)

    enqueueQueuedPrompt('stored-session-a', { text: 'halted by stop', attachments: [] })
    parkQueuedPrompts('stored-session-a')
    clearAllSessionStates()

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await new Promise(resolve => window.setTimeout(resolve, 0))

    expect(submitText).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })

  it('passes a null runtime id so submitText can resume stale background sessions by stored id', async () => {
    const runtimeMap = { current: new Map<string, string>() }
    const submitText = vi.fn(async () => true)

    enqueueQueuedPrompt('stored-session-a', { text: 'resume then send', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await waitFor(() => {
      expect(submitText).toHaveBeenCalledWith('resume then send', {
        attachments: [],
        fromQueue: true,
        sessionId: null,
        storedSessionId: 'stored-session-a'
      })
    })
  })

  it('retries a rejected background drain without waiting for another queue or busy-state change', async () => {
    vi.useFakeTimers()

    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn().mockResolvedValueOnce(false).mockResolvedValueOnce(true)

    enqueueQueuedPrompt('stored-session-a', { text: 'retry me', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await act(async () => {
      await Promise.resolve()
    })

    expect(submitText).toHaveBeenCalledTimes(1)
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(750)
      await Promise.resolve()
    })

    expect(submitText).toHaveBeenCalledTimes(2)
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(0)
  })
})
