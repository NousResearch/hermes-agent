import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { ENTRY_PREFIX } from './prepared-submissions'
import { useSubmitPrompt } from './submit'
import type { GatewayRequest } from './utils'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

beforeEach(() => {
  window.localStorage.clear()
  $sessions.set([])
  $sessionStates.set({})
  $connection.set(null)
  $activeGatewayProfile.set('default')
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

type ClientSessionState = ReturnType<typeof createClientSessionState>

// A browser origin at its storage quota refuses the pre-send journal write. The send is the user's
// request, not the cache's: it must reach prompt.submit anyway (#156129), with the refused entry
// kept in-window so a lost acknowledgement can still retry under the same submission id.
it('a full localStorage quota does not abort the send: prompt.submit still runs', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  let state: ClientSessionState = createClientSessionState()
  const submitted: Record<string, unknown>[] = []

  const setItem = window.localStorage.setItem.bind(window.localStorage)
  vi.spyOn(window.localStorage, 'setItem').mockImplementation((key: string, value: string) => {
    if (key.startsWith(ENTRY_PREFIX)) { throw new DOMException('The quota has been exceeded.', 'QuotaExceededError') }
    setItem(key, value)
  })
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'prompt.submit') {
      submitted.push(params ?? {})

      return { submission_id: params?.submission_id, session_id: 'runtime-a', status: 'started' } as never
    }

    return {} as never
  })

  const deps = submitDeps(requestGateway as GatewayRequest, updater => (state = updater(state)))
  const { result } = renderHook(() => useSubmitPrompt(deps as never))
  await act(async () => {
    expect(await result.current('look at this')).toBe(true)
  })

  expect(submitted).toHaveLength(1)
  expect(warn).toHaveBeenCalledWith('[prepared-submission-unstored]', expect.objectContaining({ name: 'QuotaExceededError' }))
  expect(state.messages.filter(message => message.role === 'user')).toHaveLength(1)
  expect(state.messages.some(message => message.error)).toBe(false)
})

function submitDeps(
  requestGateway: GatewayRequest,
  update: (updater: (s: ClientSessionState) => ClientSessionState) => ClientSessionState
) {
  return {
    activeSessionIdRef: { current: 'runtime-a' as string | null },
    selectedStoredSessionIdRef: { current: 'stored-a' as string | null },
    busyRef: { current: false },
    copy: en.desktop,
    createBackendSessionForSend: vi.fn(async () => null),
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: () => 'runtime-a',
    getRouteToken: () => 'same-route',
    requestGateway,
    runtimeIdByStoredSessionIdRef: { current: new Map([['stored-a', 'runtime-a']]) },
    resumeStoredSession: vi.fn(),
    syncAttachmentsForSubmit: vi.fn(async (sessionId: string) => ({ sessionId, attachments: [] })),
    updateSessionState: vi.fn((_sid: string, updater: (s: ClientSessionState) => ClientSessionState) => update(updater))
  }
}
