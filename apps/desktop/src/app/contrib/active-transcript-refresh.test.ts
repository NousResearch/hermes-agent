import type { MutableRefObject } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { SessionMessage, SessionMessagesResponse } from '@/hermes'
import type { SessionInfo } from '@/types/hermes'

import type { ClientSessionState } from '../types'

import { advanceTranscriptScope, refreshActiveTranscript } from './active-transcript-refresh'

const ref = <T>(current: T): MutableRefObject<T> => ({ current })

const idleState = (storedSessionId = 'stored-a'): ClientSessionState =>
  ({
    storedSessionId,
    messages: [],
    busy: false,
    awaitingResponse: false,
    streamId: null
  }) as unknown as ClientSessionState

const message = (content: string): SessionMessage => ({ content, role: 'user', timestamp: 1 })
const response = (messages: SessionMessage[]): SessionMessagesResponse => ({ messages, session_id: 'stored-a' })
const stored = (id = 'stored-a'): SessionInfo => ({ id, profile: 'san' }) as SessionInfo

function harness() {
  const activeSessionIdRef = ref<null | string>('runtime-a')
  const busyRef = ref(false)
  const generationRef = ref(0)
  const selectedStoredSessionIdRef = ref<null | string>('stored-a')
  const sessionStateByRuntimeIdRef = ref(new Map([['runtime-a', idleState()]]))
  const signatureRef = ref(new Map<string, string>())
  const findStoredSession = vi.fn(() => stored())

  const updateSessionState = vi.fn(
    (runtimeSessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => {
      const previous = sessionStateByRuntimeIdRef.current.get(runtimeSessionId)!
      const next = updater(previous)
      sessionStateByRuntimeIdRef.current.set(runtimeSessionId, next)

      return next
    }
  )

  return {
    activeSessionIdRef,
    busyRef,
    findStoredSession,
    generationRef,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    signatureRef,
    updateSessionState
  }
}

describe('active transcript refresh', () => {
  it('publishes a changed persisted transcript into its idle owning runtime', async () => {
    const h = harness()

    await refreshActiveTranscript({
      ...h,
      loadMessages: vi.fn().mockResolvedValue(response([message('remote')]))
    })

    expect(h.updateSessionState).toHaveBeenCalledTimes(1)
    expect(h.sessionStateByRuntimeIdRef.current.get('runtime-a')?.messages).toHaveLength(1)
    expect(h.signatureRef.current.size).toBe(1)
  })

  it('rejects an old A response after an A to B to A scope transition', async () => {
    const h = harness()
    const scopeKeyRef = ref('remote-a:san:stored-a:runtime-a')
    let resolve: ((messages: SessionMessagesResponse) => void) | undefined

    const loadMessages = vi.fn(
      () =>
        new Promise<SessionMessagesResponse>(res => {
          resolve = res
        })
    )

    const pending = refreshActiveTranscript({ ...h, loadMessages })

    advanceTranscriptScope(
      { generationRef: h.generationRef, scopeKeyRef },
      'remote-a:san:stored-b:runtime-b'
    )
    advanceTranscriptScope(
      { generationRef: h.generationRef, scopeKeyRef },
      'remote-a:san:stored-a:runtime-a'
    )
    resolve?.(response([message('stale')]))
    await pending

    expect(h.updateSessionState).not.toHaveBeenCalled()
    expect(h.signatureRef.current.size).toBe(0)
  })

  it('rejects a response when a local turn starts while the request is pending', async () => {
    const h = harness()
    let resolve: ((messages: SessionMessagesResponse) => void) | undefined

    const loadMessages = vi.fn(
      () =>
        new Promise<SessionMessagesResponse>(res => {
          resolve = res
        })
    )

    const pending = refreshActiveTranscript({ ...h, loadMessages })
    h.busyRef.current = true
    h.sessionStateByRuntimeIdRef.current.set('runtime-a', { ...idleState(), busy: true })
    resolve?.(response([message('stale')]))
    await pending

    expect(h.updateSessionState).not.toHaveBeenCalled()
  })

  it('keeps the runtime untouched when the persisted signature did not change', async () => {
    const h = harness()
    const latest = [message('same')]

    await refreshActiveTranscript({ ...h, loadMessages: vi.fn().mockResolvedValue(response(latest)) })
    h.updateSessionState.mockClear()
    await refreshActiveTranscript({ ...h, loadMessages: vi.fn().mockResolvedValue(response(latest)) })

    expect(h.updateSessionState).not.toHaveBeenCalled()
  })
})
