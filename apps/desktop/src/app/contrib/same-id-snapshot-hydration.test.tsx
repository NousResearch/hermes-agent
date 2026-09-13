import { useStore } from '@nanostores/react'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { useMemo, useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { clearSingleFlightSessionResumeState } from '@/app/session/hooks/use-prompt-actions/single-flight-resume'
import { useSessionActions } from '@/app/session/hooks/use-session-actions'
import { useSessionStateCache } from '@/app/session/hooks/use-session-state-cache'
import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'
import {
  $activeSessionId,
  $messages,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { $sessionStates, $sessionTiles, clearAllSessionStates, sessionTileDelegate } from '@/store/session-states'
import { $todosBySession, clearSessionTodos } from '@/store/todos'
import { $transcriptTailBySessionId, transcriptTailState } from '@/store/transcript-tail'
import { deferred } from '@/test/deferred'
import type { SessionInfo, SessionMessage, SessionMessagesResponse, SessionResumeResponse } from '@/types/hermes'

import { useSessionTileDelegate } from './hooks/use-session-tile-delegate'
import { createPostTurnHydrator } from './post-turn-hydration'

vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))
vi.mock('@/lib/complete-sound', () => ({ playCompleteSound: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))

const SID = 'same-runtime'
const STORED = 'same-stored'

const oldRows: SessionMessage[] = [
  { id: 1, role: 'user', content: 'old question', timestamp: 1 },
  { id: 2, role: 'assistant', content: 'old answer', timestamp: 2 }
]

const newRows: SessionMessage[] = [...oldRows, { id: 3, role: 'system', content: 'canonical snapshot', timestamp: 3 }]

function reset() {
  clearSingleFlightSessionResumeState()
  clearAllSessionStates()
  $sessionTiles.set([])
  setSessions([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setMessages([])
  setBusy(false)
  setAwaitingResponse(false)
  clearSessionTodos(SID)
  $transcriptTailBySessionId.set({})
  setApiRequestConnection(null)
  setApiRequestProfile(null)
}

beforeEach(() => {
  window.localStorage.clear()
  reset()
})
afterEach(() => {
  cleanup()
  reset()
  vi.restoreAllMocks()
})

it.each(['active', 'tile'] as const)(
  'rejects older REST hydration after a same-ID %s snapshot without dropping its accepted tail',
  async surface => {
    const staleRead = deferred<SessionMessagesResponse>()
    const snapshot = deferred<SessionResumeResponse>()
    const api = vi.fn(() => staleRead.promise)
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

    const requestGateway = vi.fn(async (method: string) => {
      if (method !== 'session.resume' && method !== 'session.activate') {
        throw new Error(`Unexpected RPC: ${method}`)
      }

      return snapshot.promise as never
    })

    const foreground = surface === 'active' ? SID : 'other-runtime'
    setActiveSessionId(foreground)
    setSelectedStoredSessionId(surface === 'active' ? STORED : 'other-stored')
    setSessions([{ id: STORED, message_count: 2, source: 'desktop', title: 'Fixture' } as SessionInfo])

    if (surface === 'tile') {
      $sessionTiles.set([{ runtimeId: SID, storedSessionId: STORED }] as never)
    }

    const hook = renderHook(() => {
      const activeSessionId = useStore($activeSessionId)
      const busyRef = useRef(false)

      const cache = useSessionStateCache({
        activeSessionId,
        busyRef,
        selectedStoredSessionId: surface === 'active' ? STORED : 'other-stored',
        setAwaitingResponse,
        setBusy,
        setMessages
      })

      const {
        activeSessionIdRef,
        selectedStoredSessionIdRef,
        sessionStateByRuntimeIdRef,
        runtimeIdByStoredSessionIdRef,
        updateSessionState
      } = cache

      const hydrate = useMemo(
        () =>
          createPostTurnHydrator({
            activeSessionIdRef,
            selectedStoredSessionIdRef,
            sessionStateByRuntimeIdRef,
            runtimeIdByStoredSessionIdRef,
            updateSessionState
          }),
        [
          activeSessionIdRef,
          selectedStoredSessionIdRef,
          sessionStateByRuntimeIdRef,
          runtimeIdByStoredSessionIdRef,
          updateSessionState
        ]
      )

      const actions = useSessionActions({
        ...cache,
        activeSessionId,
        busyRef,
        creatingSessionRef: useRef(false),
        selectedStoredSessionId: surface === 'active' ? STORED : 'other-stored',
        getRouteToken: () => 'fixture',
        getRoutedStoredSessionId: () => STORED,
        navigate: vi.fn(),
        requestGateway
      })

      useSessionTileDelegate({
        ...cache,
        requestGateway,
        archiveSession: vi.fn(),
        branchStoredSession: vi.fn(),
        removeSession: vi.fn(),
        executeSlashCommand: vi.fn()
      })

      return { cache, hydrate, actions }
    })

    await act(async () => {
      hook.result.current.cache.updateSessionState(
        SID,
        state => ({ ...state, messages: toChatMessages(oldRows) }),
        STORED
      )
    })
    const stale = hook.result.current.hydrate(1, STORED, SID)
    expect(api).toHaveBeenCalledOnce()
    let resume!: Promise<unknown>
    await act(async () => {
      resume =
        surface === 'active'
          ? hook.result.current.actions.resumeSession(STORED, true, undefined, { authoritativeSnapshot: true })
          : sessionTileDelegate()!.resumeTile(STORED, { authoritativeSnapshot: true })
    })
    await waitFor(() => expect(requestGateway).toHaveBeenCalledOnce())
    expect(requestGateway).toHaveBeenCalledWith(
      'session.resume',
      expect.objectContaining({ session_id: STORED, omit_messages: false })
    )

    // An accepted user tail can land while the authoritative read is pending.
    const accepted = {
      id: 'user-accepted',
      role: 'user' as const,
      pending: true,
      parts: [textPart('accepted next prompt')]
    }

    await act(async () => {
      hook.result.current.cache.updateSessionState(SID, state => ({
        ...state,
        messages: [...state.messages, accepted]
      }))
      snapshot.resolve({
        session_id: SID,
        resumed: STORED,
        messages: newRows,
        running: false,
        info: {}
      } as SessionResumeResponse)
      await resume
    })
    const states = hook.result.current.cache.sessionStateByRuntimeIdRef.current
    const authoritative = states.get(SID)!
    expect(authoritative.messages.map(chatMessageText)).toContain('canonical snapshot')
    expect.soft(authoritative.messages).toContainEqual(accepted)
    const view = $messages.get()
    const tileMirror = $sessionStates.get()[SID]
    const todos = $todosBySession.get()
    const scope = { connectionId: 'local', profile: 'default' }
    const tail = transcriptTailState(STORED, scope)
    await act(async () => {
      staleRead.resolve({ session_id: STORED, messages: oldRows })
      await stale
    })
    expect.soft(states.get(SID)).toBe(authoritative)
    expect.soft($sessionStates.get()[SID]).toBe(tileMirror)
    expect.soft($messages.get()).toBe(view)
    expect.soft($todosBySession.get()).toBe(todos)
    expect.soft(transcriptTailState(STORED, scope)).toBe(tail)
    expect($activeSessionId.get()).toBe(foreground)

    // The boundary must reject old reads, not permanently disable the hydrator.
    api.mockResolvedValueOnce({
      session_id: STORED,
      messages: [...newRows, { id: 4, role: 'system', content: 'later persisted history' }]
    })
    await act(async () => {
      await hook.result.current.hydrate(1, STORED, SID)
    })
    expect(states.get(SID)!.messages.map(chatMessageText)).toContain('later persisted history')
    expect(requestGateway).toHaveBeenCalledOnce()

    // Ordinary refresh/reopen still layers persisted history. It is not a new
    // canonical authority boundary merely because it writes the same cache.
    const epoch = states.get(SID)!.transcriptAuthorityEpoch
    api.mockResolvedValueOnce({ session_id: STORED, messages: newRows })
    await act(async () => {
      if (surface === 'active') {
        await hook.result.current.actions.resumeSession(STORED, true)
      } else {
        await sessionTileDelegate()!.resumeTile(STORED, { refreshTranscript: true })
      }
    })
    expect(states.get(SID)!.transcriptAuthorityEpoch).toBe(epoch)
    expect(requestGateway.mock.calls.map(([method]) => method)).toEqual(
      surface === 'active' ? ['session.resume', 'session.activate'] : ['session.resume']
    )
  }
)
