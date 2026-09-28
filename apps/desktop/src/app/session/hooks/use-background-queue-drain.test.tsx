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
  parkQueuedPrompts
} from '@/store/composer-queue'
import { $notifications, clearNotifications } from '@/store/notifications'
import {
  $sessions,
  _resetSessionOwnerHintsForTests,
  setSessionOwnerHint,
  setSessionProfilesTruncated,
  setSessions,
  setSessionsLoading
} from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { useBackgroundQueueDrain } from './use-background-queue-drain'
import type { SubmitTextOptions } from './use-prompt-actions/utils'

const lineageSession = (over: Partial<SessionInfo>): SessionInfo =>
  ({
    archived: false,
    cwd: null,
    ended_at: null,
    id: 'live',
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: null,
    tool_call_count: 0,
    ...over
  }) as SessionInfo

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
    _resetSessionOwnerHintsForTests()
    // The queue store merges over live localStorage on save (cross-window sync,
    // #46732) — stale persisted entries from an earlier test would be adopted
    // into the atom and drained here as if they were fresh queue state.
    window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
    // Production drain waits for the sidebar list. Tests that assert drain
    // behavior are post-load unless they opt into the loading gate.
    setSessionsLoading(false)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.useRealTimers()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    $sessions.set([])
    setSessionsLoading(true)
    clearNotifications()
    clearAllSessionStates()
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

  it('treats a tip working id as busy for a root queue key via lineage', async () => {
    // Queue keys use the lineage root (resolveComposerSessionKey) while
    // $workingSessionIds may hold the compression tip — strict equality misses.
    const runtimeMap = { current: new Map([['root-a', 'rt-tip-a']]) }
    const submitText = vi.fn(async () => true)

    setSessions([lineageSession({ id: 'tip-a', _lineage_root_id: 'root-a' })])
    enqueueQueuedPrompt('root-a', { text: 'wait for tip turn', attachments: [] })
    publishSessionState('rt-tip-a', { ...createClientSessionState('tip-a'), busy: true })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await new Promise(resolve => window.setTimeout(resolve, 0))

    expect(submitText).not.toHaveBeenCalled()
    expect(getQueuedPrompts('root-a')).toHaveLength(1)
  })

  it('leaves a root queue to ChatBar when the selected id is the compression tip', async () => {
    const runtimeMap = { current: new Map([['root-a', 'rt-tip-a']]) }
    const submitText = vi.fn(async () => true)

    setSessions([lineageSession({ id: 'tip-a', _lineage_root_id: 'root-a' })])
    enqueueQueuedPrompt('root-a', { text: 'visible after tip select', attachments: [] })
    clearAllSessionStates()

    render(<Harness runtimeMap={runtimeMap} selectedStoredSessionId="tip-a" submitText={submitText} />)

    await new Promise(resolve => window.setTimeout(resolve, 0))

    expect(submitText).not.toHaveBeenCalled()
    expect(getQueuedPrompts('root-a')).toHaveLength(1)
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

  it('does not drain restored queues while the session list is still loading', async () => {
    vi.useFakeTimers()
    setSessionsLoading(true)

    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => true)

    enqueueQueuedPrompt('stored-session-a', { text: 'wait for session list', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    // Four 750ms retries is what used to burn the drain budget and toast on boot.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(750 * 4)
      await Promise.resolve()
    })

    expect(submitText).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })

  it('drains a restored background queue once the session list finishes loading', async () => {
    setSessionsLoading(true)

    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => true)

    enqueueQueuedPrompt('stored-session-a', { text: 'send after load', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await new Promise(resolve => window.setTimeout(resolve, 0))
    expect(submitText).not.toHaveBeenCalled()

    setSessionsLoading(false)

    await waitFor(() => {
      expect(submitText).toHaveBeenCalledWith('send after load', {
        attachments: [],
        fromQueue: true,
        sessionId: 'rt-session-a',
        storedSessionId: 'stored-session-a'
      })
    })

    await waitFor(() => expect(getQueuedPrompts('stored-session-a')).toHaveLength(0))
  })

  it("drops a gone session's queued prompt quietly at drain exhaustion instead of erroring (#98015)", async () => {
    vi.useFakeTimers()

    // The session list has settled and no row answers to the queued session:
    // the conversation is gone from this backend (reaped runtime whose stored
    // resume refuses — the post-restart shape the issue reports).
    setSessions([])
    const runtimeMap = { current: new Map<string, string>() }
    const submitText = vi.fn(async () => false)

    enqueueQueuedPrompt('stored-session-a', { text: 'never sends', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await act(async () => {
      await Promise.resolve()
    })

    for (let attempt = 1; attempt < MAX_AUTO_DRAIN_ATTEMPTS; attempt++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(750)
      })
    }

    expect(submitText).toHaveBeenCalledTimes(MAX_AUTO_DRAIN_ATTEMPTS)

    // The entry is dropped and the notice is a quiet info, not the old
    // "message not sent" error banner.
    await act(async () => {
      await Promise.resolve()
    })
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(0)
    const stuck = $notifications.get().find(n => n.id === 'composer-background-queue-stuck-stored-session-a')
    expect(stuck?.kind).toBe('info')
  })

  it('keeps the queued prompt for a session the loaded list still knows, with a quiet notice (#98015)', async () => {
    vi.useFakeTimers()

    setSessions([lineageSession({ id: 'stored-session-a' })])
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => false)

    enqueueQueuedPrompt('stored-session-a', { text: 'retry from the panel', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await act(async () => {
      await Promise.resolve()
    })

    for (let attempt = 1; attempt < MAX_AUTO_DRAIN_ATTEMPTS; attempt++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(750)
      })
    }

    // The conversation exists — the entry survives for a manual send, and the
    // notice is downgraded from the error banner.
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
    const stuck = $notifications.get().find(n => n.id === 'composer-background-queue-stuck-stored-session-a')
    expect(stuck?.kind).toBe('info')
  })

  it('keeps the queued prompt when the session is reachable but its owner hint is ambiguous (#122083 review)', async () => {
    vi.useFakeTimers()

    // Two routes for the same id (cloud gateway + local backend, or a profile
    // switch that re-stamped the route) make getSessionOwnerHint return
    // undefined — a POSITIVE liveness signal, not absence. Reading the
    // singular accessor as an existence test dropped the user's queued
    // prompt at drain exhaustion.
    setSessions([])
    setSessionOwnerHint('stored-session-a', { connectionId: 'conn-cloud', profile: 'default' })
    setSessionOwnerHint('stored-session-a', { connectionId: 'conn-local', profile: 'work' })
    const runtimeMap = { current: new Map<string, string>() }
    const submitText = vi.fn(async () => false)

    enqueueQueuedPrompt('stored-session-a', { text: 'still alive', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await act(async () => {
      await Promise.resolve()
    })

    for (let attempt = 1; attempt < MAX_AUTO_DRAIN_ATTEMPTS; attempt++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(750)
      })
    }

    expect(submitText).toHaveBeenCalledTimes(MAX_AUTO_DRAIN_ATTEMPTS)
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })

  it('keeps the queued prompt when the loaded list page cannot prove the session gone (#122083 review)', async () => {
    vi.useFakeTimers()

    // $sessions is one PAGE of the sidebar list: a queued session that simply
    // fell off the loaded window ($sessionProfilesTruncated) is unknown by
    // row and by hint — but "not on this page" is not "deleted". Dropping the
    // entry there destroyed real data.
    setSessionProfilesTruncated({ default: true })
    setSessions([])
    const runtimeMap = { current: new Map<string, string>() }
    const submitText = vi.fn(async () => false)

    enqueueQueuedPrompt('stored-session-a', { text: 'below the fold', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await act(async () => {
      await Promise.resolve()
    })

    for (let attempt = 1; attempt < MAX_AUTO_DRAIN_ATTEMPTS; attempt++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(750)
      })
    }

    expect(submitText).toHaveBeenCalledTimes(MAX_AUTO_DRAIN_ATTEMPTS)
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
    setSessionProfilesTruncated({})
  })

  it('does not replay the retry ladder for a queue restored after prior exhaustion (#98015)', async () => {
    vi.useFakeTimers()

    // A queue restored from localStorage with the persisted drain budget
    // already spent: no attempts, no notice — the every-boot replay of four
    // rejections plus a banner is the reported symptom.
    const runtimeMap = { current: new Map<string, string>() }
    const submitText = vi.fn(async () => true)

    setSessions([lineageSession({ id: 'stored-session-a' })])
    $queuedPromptsBySession.set({
      'stored-session-a': [
        {
          id: 'queued-restored',
          text: 'already exhausted in a previous process',
          attachments: [],
          queuedAt: 1,
          drainFailures: MAX_AUTO_DRAIN_ATTEMPTS
        }
      ]
    })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(750 * 6)
    })

    expect(submitText).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-session-a').map(e => e.id)).toEqual(['queued-restored'])
    expect($notifications.get()).toHaveLength(0)
  })
})
