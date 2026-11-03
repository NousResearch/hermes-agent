import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { en } from '@/i18n/en'
import { chatMessageText, textPart } from '@/lib/chat-messages'
import {
  $connection,
  $messages,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

import { useSessionStateCache } from '../use-session-state-cache'

import { useSubmitPrompt } from './submit'

const SID = 'local-first-input'
const INPUT = 'Prepare the weekly report'

function mount(startBeforeReceipt = false) {
  let start = () => undefined

  const request = vi.fn(async (_method: string, params: Record<string, unknown>) => {
    if (startBeforeReceipt) {
      start()
    }

    return { admission_id: 'admission-1', submission_id: params.submission_id, session_id: SID, status: 'queued' }
  })

  const hook = renderHook(() => {
    const busyRef = useRef(false)

    const cache = useSessionStateCache({
      activeSessionId: SID,
      selectedStoredSessionId: SID,
      busyRef,
      setMessages,
      setBusy,
      setAwaitingResponse
    })

    const submit = useSubmitPrompt({
      ...cache,
      busyRef,
      copy: en.desktop,
      createBackendSessionForSend: async () => null,
      getRoutedStoredSessionId: () => SID,
      getRouteToken: () => `/${SID}`,
      requestGateway: request as Parameters<typeof useSubmitPrompt>[0]['requestGateway'],
      resumeStoredSession: async () => undefined,
      syncAttachmentsForSubmit: async (sessionId, attachments) => ({ sessionId, attachments })
    })

    return { cache, submit }
  })

  act(() => {
    hook.result.current.cache.updateSessionState(SID, state => state, SID)
  })

  start = () => {
    hook.result.current.cache.updateSessionState(
      SID,
      state => ({
        ...state,
        busy: true,
        awaitingResponse: true,
        turnLive: true,
        turnStartedAt: 1234
      }),
      SID
    )
  }

  return { hook, request }
}

beforeEach(() => {
  window.localStorage.clear()
  clearAllSessionStates()
  setMessages([])
  setBusy(false)
  setAwaitingResponse(false)
  setActiveSessionId(SID)
  setSelectedStoredSessionId(SID)
  setSessions([makeSessionInfo({ id: SID, source: 'gui', message_count: 0 })])
  $connection.set({
    mode: 'local',
    connectionId: 'local',
    profile: 'default',
    baseUrl: 'http://127.0.0.1:1',
    wsUrl: 'ws://127.0.0.1:1/api/ws?native_dial=1',
    token: '',
    logs: [],
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: null
  })
})

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  setMessages([])
  setBusy(false)
  setAwaitingResponse(false)
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  $connection.set(null)
  vi.restoreAllMocks()
})

it.each([false, true])('keeps first input visible when a queued receipt races start=%s', async startBeforeReceipt => {
  const { hook, request } = mount(startBeforeReceipt)
  await act(async () => {
    expect(await hook.result.current.submit(INPUT)).toBe(true)
  })
  expect(request).toHaveBeenCalledTimes(1)
  expect(PRIMARY_SESSION_VIEW.$messages.get().map(chatMessageText)).toEqual([INPUT])
  expect(PRIMARY_SESSION_VIEW.$messagesEmpty.get()).toBe(false)
  expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)

  if (startBeforeReceipt) {
    expect(hook.result.current.cache.sessionStateByRuntimeIdRef.current.get(SID)).toMatchObject({
      turnLive: true,
      turnStartedAt: 1234
    })
  }
})

it('does not graft explicitly queued input onto the active turn', async () => {
  const { hook } = mount()
  act(() => {
    hook.result.current.cache.updateSessionState(
      SID,
      state => ({
        ...state,
        busy: true,
        turnLive: true,
        messages: [{ id: 'current-answer', role: 'assistant', parts: [textPart('Still working')] }]
      }),
      SID
    )
  })
  await act(async () => {
    expect(await hook.result.current.submit(INPUT, { fromQueue: true })).toBe(true)
  })
  expect(PRIMARY_SESSION_VIEW.$messages.get().map(chatMessageText)).toEqual(['Still working'])
  expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)
  expect($messages.get().some(message => chatMessageText(message) === INPUT)).toBe(false)
})
