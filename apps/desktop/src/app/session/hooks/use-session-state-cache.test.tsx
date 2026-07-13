import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { $petLiveSessions, resetPetLiveSessions } from '@/store/pet-live-session'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentServiceTier,
  $messages,
  $turnStartedAt,
  $workingSessions,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setTurnStartedAt
} from '@/store/session'
import { getProfileSessionValue } from '@/store/session-identity'

import { useSessionStateCache } from './use-session-state-cache'

type Cache = ReturnType<typeof useSessionStateCache>

interface HarnessProps {
  activeSessionId: string | null
  onReady: (cache: Cache) => void
  selectedStoredSessionId: string | null
}

function Harness({ activeSessionId, onReady, selectedStoredSessionId }: HarnessProps) {
  const busyRef: MutableRefObject<boolean> = { current: false }

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId,
    setAwaitingResponse: () => undefined,
    setBusy: () => undefined,
    setMessages: () => undefined
  })

  onReady(cache)

  return null
}

describe('useSessionStateCache — per-session turn timer', () => {
  beforeEach(() => {
    // The view-sync flush runs on a real rAF in the browser path; in jsdom we
    // want it synchronous so the global mirror is observable immediately. The
    // hook closes over `window.requestAnimationFrame`, so stub that exact ref.
    // Return null (not a handle) so the hook's `viewSyncRafRef.current = rAF(...)`
    // assignment doesn't overwrite the null the synchronous callback just set —
    // otherwise the ref reads truthy and the NEXT sync is suppressed (a real
    // browser returns a handle but runs the callback async, so this race is a
    // test-only artifact of firing synchronously).
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb: FrameRequestCallback) => {
      cb(0)

      return null as unknown as number
    })
    setTurnStartedAt(null)
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentReasoningEffort('')
    setCurrentServiceTier('')
    setCurrentFastMode(false)
    $activeGatewayProfile.set('default')
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    setTurnStartedAt(null)
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentReasoningEffort('')
    setCurrentServiceTier('')
    setCurrentFastMode(false)
  })

  it("keeps a background session's running turn clock and never mirrors it to the view", () => {
    let cache!: Cache
    // Active session is "fg-runtime"; the turn starts on the BACKGROUND session.
    render(<Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />)

    const startedAt = 1_700_000_000_000

    act(() => {
      cache.updateSessionState('bg-runtime', state => ({ ...state, busy: true, turnStartedAt: startedAt }), 'bg-stored')
    })

    // The background session's own cache entry holds the clock...
    expect(getProfileSessionValue(cache.sessionStateByRuntimeIdRef.current, 'default', 'bg-runtime')?.turnStartedAt).toBe(
      startedAt
    )
    // ...but the global atom (statusbar timer) is untouched — a background turn
    // must not drive the foreground timer.
    expect($turnStartedAt.get()).toBeNull()
  })

  it("mirrors the focused session's turn clock into the global atom on view-sync", () => {
    let cache!: Cache
    render(<Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />)

    const startedAt = 1_700_000_111_000

    // A turn on the ACTIVE session stages into the view; the flush mirrors its
    // turnStartedAt into the global atom the statusbar reads.
    act(() => {
      cache.updateSessionState('fg-runtime', state => ({ ...state, busy: true, turnStartedAt: startedAt }), 'fg-stored')
    })

    expect($turnStartedAt.get()).toBe(startedAt)
  })

  it('clears the global clock when the focused turn ends', () => {
    let cache!: Cache
    render(<Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />)

    act(() => {
      cache.updateSessionState(
        'fg-runtime',
        state => ({ ...state, busy: true, turnStartedAt: 1_700_000_222_000 }),
        'fg-stored'
      )
    })
    expect($turnStartedAt.get()).toBe(1_700_000_222_000)

    act(() => {
      cache.updateSessionState('fg-runtime', state => ({ ...state, busy: false, turnStartedAt: null }))
    })
    expect($turnStartedAt.get()).toBeNull()
  })

  it('mirrors the focused session model metadata when switching from a cached session', () => {
    let cache!: Cache

    const { rerender } = render(
      <Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />
    )

    act(() => {
      cache.updateSessionState(
        'bg-runtime',
        state => ({
          ...state,
          fast: true,
          model: 'anthropic/claude-opus-4.8',
          provider: 'anthropic',
          reasoningEffort: 'high',
          serviceTier: 'priority'
        }),
        'bg-stored'
      )
    })

    // Background metadata is cached but must not bleed into the visible statusbar.
    expect($currentModel.get()).toBe('')
    expect($currentReasoningEffort.get()).toBe('')
    expect($currentFastMode.get()).toBe(false)

    rerender(<Harness activeSessionId="bg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="bg-stored" />)

    const bgState = getProfileSessionValue(cache.sessionStateByRuntimeIdRef.current, 'default', 'bg-runtime')
    expect(bgState).toBeTruthy()

    act(() => {
      cache.syncSessionStateToView('bg-runtime', bgState!)
    })

    expect($currentModel.get()).toBe('anthropic/claude-opus-4.8')
    expect($currentProvider.get()).toBe('anthropic')
    expect($currentReasoningEffort.get()).toBe('high')
    expect($currentServiceTier.get()).toBe('priority')
    expect($currentFastMode.get()).toBe(true)
  })

  it('clears stale model metadata when the newly focused session has no cached value', () => {
    setCurrentModel('previous-model')
    setCurrentProvider('previous-provider')
    setCurrentReasoningEffort('high')
    setCurrentServiceTier('priority')
    setCurrentFastMode(true)

    let cache!: Cache

    const { rerender } = render(
      <Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />
    )

    act(() => {
      cache.updateSessionState('bg-runtime', state => ({ ...state }), 'bg-stored')
    })

    rerender(<Harness activeSessionId="bg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="bg-stored" />)

    const bgState = getProfileSessionValue(cache.sessionStateByRuntimeIdRef.current, 'default', 'bg-runtime')
    expect(bgState).toBeTruthy()

    act(() => {
      cache.syncSessionStateToView('bg-runtime', bgState!)
    })

    expect($currentModel.get()).toBe('')
    expect($currentProvider.get()).toBe('')
    expect($currentReasoningEffort.get()).toBe('')
    expect($currentServiceTier.get()).toBe('')
    expect($currentFastMode.get()).toBe(false)
  })
})

describe('useSessionStateCache — profile identity', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    $activeGatewayProfile.set('default')
    $messages.set([])
    $workingSessions.set([])
    resetPetLiveSessions()
  })

  it('stores the same runtime and stored ids independently across profiles', () => {
    let cache!: Cache
    render(<ViewHarness activeSessionId="same-runtime" onReady={c => (cache = c)} />)

    act(() => {
      cache.updateSessionState(
        'same-runtime',
        state => ({ ...state, messages: [userMessage('default-message', 'default')] }),
        'same-stored',
        'default'
      )
      cache.updateSessionState(
        'same-runtime',
        state => ({ ...state, messages: [userMessage('work-message', 'work')] }),
        'same-stored',
        'work'
      )
    })

    expect(cache.sessionStateByRuntimeIdRef.current.size).toBe(2)
    expect(cache.runtimeIdByStoredSessionIdRef.current.size).toBe(2)
    expect(getProfileSessionValue(cache.runtimeIdByStoredSessionIdRef.current, 'default', 'same-stored')).toBe(
      'same-runtime'
    )
    expect(getProfileSessionValue(cache.runtimeIdByStoredSessionIdRef.current, 'work', 'same-stored')).toBe(
      'same-runtime'
    )
    expect(
      [...cache.sessionStateByRuntimeIdRef.current.values()].map(state => state.messages[0]?.id).sort()
    ).toEqual(['default-message', 'work-message'])
  })

  it('syncs canonical cache updates into safe live snapshots and re-homes exact active idle on profile/session switch', () => {
    let cache!: Cache
    $activeGatewayProfile.set('default')
    const { rerender } = render(<ViewHarness activeSessionId="default-runtime" onReady={c => (cache = c)} />)

    act(() => {
      cache.updateSessionState(
        'default-runtime',
        current => ({ ...current, busy: true, awaitingResponse: true, turnStartedAt: 10 }),
        'default-stored',
        'default'
      )
      cache.updateSessionState(
        'work-runtime',
        current => ({ ...current, busy: false, awaitingResponse: false, turnStartedAt: null }),
        'work-stored',
        'work'
      )
    })

    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({ profile: 'default', runtimeSessionId: 'default-runtime', busy: true })
    ])

    act(() => {
      $activeGatewayProfile.set('work')
      rerender(<ViewHarness activeSessionId="work-runtime" onReady={c => (cache = c)} />)
    })

    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({
        profile: 'default',
        runtimeSessionId: 'default-runtime',
        busy: true
      }),
      expect.objectContaining({
        profile: 'work',
        runtimeSessionId: 'work-runtime',
        busy: false,
        storedSessionId: 'work-stored'
      })
    ])
  })

  it('publishes only the matching active profile when runtime ids collide', () => {
    let cache!: Cache
    $activeGatewayProfile.set('default')
    render(<ViewHarness activeSessionId="same-runtime" onReady={c => (cache = c)} />)

    act(() => {
      cache.updateSessionState(
        'same-runtime',
        state => ({ ...state, busy: false, messages: [userMessage('work-message', 'work')] }),
        'same-stored',
        'work'
      )
    })
    expect($messages.get()).toEqual([])

    act(() => {
      cache.updateSessionState(
        'same-runtime',
        state => ({ ...state, busy: false, messages: [userMessage('default-message', 'default')] }),
        'same-stored',
        'default'
      )
    })
    expect($messages.get().map(message => message.id)).toEqual(['default-message'])
  })

  it('watchdog healing clears only the matching profile cache entry', () => {
    vi.useFakeTimers()
    let cache!: Cache
    render(<ViewHarness activeSessionId="same-runtime" onReady={c => (cache = c)} />)

    act(() => {
      cache.updateSessionState(
        'same-runtime',
        state => ({ ...state, awaitingResponse: true, busy: true }),
        'same-stored',
        'default'
      )
      vi.advanceTimersByTime(1)
      cache.updateSessionState(
        'same-runtime',
        state => ({ ...state, awaitingResponse: true, busy: true }),
        'same-stored',
        'work'
      )
      vi.advanceTimersByTime(8 * 60 * 1000 - 1)
    })

    expect(getProfileSessionValue(cache.sessionStateByRuntimeIdRef.current, 'default', 'same-runtime')?.busy).toBe(
      false
    )
    expect(getProfileSessionValue(cache.sessionStateByRuntimeIdRef.current, 'work', 'same-runtime')?.busy).toBe(true)
  })
})

function userMessage(id: string, text: string): ChatMessage {
  return { id, role: 'user', parts: [{ type: 'text', text }] }
}

function assistantText(id: string, text: string): ChatMessage {
  return { id, role: 'assistant', parts: [{ type: 'text', text }] }
}

function assistantError(id: string, error: string): ChatMessage {
  return { id, role: 'assistant', parts: [], error, pending: false }
}

interface ViewHarnessProps {
  activeSessionId: string | null
  onReady: (cache: Cache) => void
}

function ViewHarness({ activeSessionId, onReady }: ViewHarnessProps) {
  const busyRef: MutableRefObject<boolean> = { current: false }

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId: null,
    setAwaitingResponse: () => undefined,
    setBusy: () => undefined,
    // Wire the published view back into the real $messages atom the flush
    // reads from, so the round-trip matches production.
    setMessages: messages => $messages.set(messages)
  })

  onReady(cache)

  return null
}

describe('useSessionStateCache — cross-thread error isolation', () => {
  afterEach(() => {
    cleanup()
    $messages.set([])
  })

  it('does not leak a failed turn into another thread on switch', () => {
    $messages.set([])
    let cache!: Cache
    const { rerender } = render(<ViewHarness activeSessionId="thread-A" onReady={c => (cache = c)} />)

    // Thread A ends its turn with an out-of-funds error and is on screen.
    act(() => {
      cache.updateSessionState(
        'thread-A',
        state => ({
          ...state,
          busy: false,
          messages: [userMessage('user-a', 'do the thing'), assistantError('assistant-a-error', 'Out of funds')]
        }),
        'stored-A'
      )
    })

    expect($messages.get().some(message => message.error === 'Out of funds')).toBe(true)

    // Switch to thread B (which completed cleanly). Its cached state syncs to
    // the view while $messages still holds thread A's transcript.
    rerender(<ViewHarness activeSessionId="thread-B" onReady={c => (cache = c)} />)
    act(() => {
      cache.updateSessionState(
        'thread-B',
        state => ({
          ...state,
          busy: false,
          messages: [userMessage('user-b', 'hello'), assistantText('assistant-b', 'hi there')]
        }),
        'stored-B'
      )
    })

    expect($messages.get().map(message => message.id)).toEqual(['user-b', 'assistant-b'])
    expect($messages.get().some(message => message.error === 'Out of funds')).toBe(false)
  })

  it('still preserves a same-session local error a heartbeat dropped', () => {
    $messages.set([])
    let cache!: Cache
    render(<ViewHarness activeSessionId="thread-A" onReady={c => (cache = c)} />)

    // First paint establishes thread A as the on-screen session.
    act(() => {
      cache.updateSessionState(
        'thread-A',
        state => ({ ...state, busy: false, messages: [userMessage('user-a', 'do the thing')] }),
        'stored-A'
      )
    })

    // A local error lands in the view (e.g. failAssistantMessage wrote it).
    $messages.set([userMessage('user-a', 'do the thing'), assistantError('assistant-a-error', 'OpenRouter 403')])

    // A later same-session heartbeat carries cached state that lost the error.
    act(() => {
      cache.updateSessionState('thread-A', state => ({
        ...state,
        busy: false,
        messages: [userMessage('user-a', 'do the thing')]
      }))
    })

    expect($messages.get().some(message => message.error === 'OpenRouter 403')).toBe(true)
  })
})
