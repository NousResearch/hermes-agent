import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { storedJournal } from './prepared-submissions-test-utils'
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

// TUI-26: a Desktop viewer of a LOCAL session whose voice chat another viewer (Ink) turned on types
// a bare "stop". The owner ends that voice chat and admits nothing: `{voice_stopped: true}`. The send
// is settled (no error bubble, no retry identity left), and its bubble goes, since no turn holds it.
it('a voice_stopped receipt settles the send with no turn, no error and no retained retry', async () => {
  let state: ClientSessionState = createClientSessionState()

  const requestGateway = vi.fn(
    async (_method: string, params?: Record<string, unknown>) =>
      ({ voice_stopped: true, session_id: 'runtime-a', submission_id: params?.submission_id }) as never
  )

  const deps = {
    activeSessionIdRef: { current: 'runtime-a' as string | null },
    selectedStoredSessionIdRef: { current: 'stored-a' as string | null },
    busyRef: { current: false },
    copy: en.desktop,
    createBackendSessionForSend: vi.fn(async () => null),
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: () => 'runtime-a',
    getRouteToken: () => 'same-route',
    requestGateway: requestGateway as GatewayRequest,
    runtimeIdByStoredSessionIdRef: { current: new Map([['stored-a', 'runtime-a']]) },
    resumeStoredSession: vi.fn(),
    syncAttachmentsForSubmit: vi.fn(async (sessionId: string) => ({ sessionId, attachments: [] })),
    updateSessionState: vi.fn((_sid: string, updater: (s: ClientSessionState) => ClientSessionState) => {
      state = updater(state)

      return state
    })
  }

  const { result } = renderHook(() => useSubmitPrompt(deps as never))
  await act(async () => {
    expect(await result.current('stop', { submission_id: 'stop-1' })).toBe(true)
  })

  expect(requestGateway.mock.calls.filter(([method]) => method === 'prompt.submit')).toHaveLength(1)
  expect(state.messages.filter(message => message.role === 'user')).toHaveLength(0)
  expect(state.messages.some(message => message.error)).toBe(false)
  expect(state.busy).toBe(false)
  expect(state.awaitingResponse).toBe(false)
  expect(storedJournal()).toEqual({})
})

// Greptile on TUI-26: the sidecar refuses `submission_id` (4000, version skew), Desktop retries the
// send identityless, and `_submit_to_ledger_owner` answers that legacy write with the same
// `{voice_stopped: true}`. The send settles there too: no turn will ever complete it.
it('a voice_stopped receipt on the identityless legacy fallback settles the send too', async () => {
  let state: ClientSessionState = createClientSessionState()

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'prompt.submit' && params && 'submission_id' in params) {
      throw Object.assign(new Error('unknown params: submission_id'), { code: 4000 })
    }

    return { voice_stopped: true } as never
  })

  const deps = {
    activeSessionIdRef: { current: 'runtime-a' as string | null },
    selectedStoredSessionIdRef: { current: 'stored-a' as string | null },
    busyRef: { current: false },
    copy: en.desktop,
    createBackendSessionForSend: vi.fn(async () => null),
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: () => 'runtime-a',
    getRouteToken: () => 'same-route',
    requestGateway: requestGateway as GatewayRequest,
    runtimeIdByStoredSessionIdRef: { current: new Map([['stored-a', 'runtime-a']]) },
    resumeStoredSession: vi.fn(),
    syncAttachmentsForSubmit: vi.fn(async (sessionId: string) => ({ sessionId, attachments: [] })),
    updateSessionState: vi.fn((_sid: string, updater: (s: ClientSessionState) => ClientSessionState) => {
      state = updater(state)

      return state
    })
  }

  const { result } = renderHook(() => useSubmitPrompt(deps as never))
  await act(async () => {
    expect(await result.current('stop', { submission_id: 'stop-2' })).toBe(true)
  })

  const submits = requestGateway.mock.calls.filter(([method]) => method === 'prompt.submit')
  expect(submits).toHaveLength(2)
  expect(submits[1][1]).not.toHaveProperty('submission_id')
  expect(state.messages.filter(message => message.role === 'user')).toHaveLength(0)
  expect(state.messages.some(message => message.error)).toBe(false)
  expect(state.busy).toBe(false)
  expect(state.awaitingResponse).toBe(false)
})
