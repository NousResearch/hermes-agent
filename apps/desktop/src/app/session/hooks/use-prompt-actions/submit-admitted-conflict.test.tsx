import { JsonRpcGatewayError } from '@hermes/shared'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { enqueueQueuedPrompt, getQueuedPrompts, queuedSubmissionId, updateQueuedPrompt } from '@/store/composer-queue'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

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

// An attachment send that reached the owner but whose landed check missed comes back as text under
// its original submission id. Enter on it is refused `admission_conflict` with `field: request_id`
// (that id is already admitted with another payload): it was delivered, so it settles with no error
// and no retained retry. Any other `admission_conflict` (route, binding) stays a visible failure.
it('a retry refused because its id is already admitted settles as delivered; other conflicts still fail', async () => {
  let state: ClientSessionState = createClientSessionState()
  let refusal: Record<string, unknown> = {}

  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'prompt.submit') {
      throw new JsonRpcGatewayError('admission_conflict', { code: 4001, data: refusal })
    }

    return {} as never
  })

  const deps = submitDeps(requestGateway as GatewayRequest, updater => (state = updater(state)))
  const { result } = renderHook(() => useSubmitPrompt(deps as never))

  refusal = {
    reason: 'admission_conflict',
    field: 'request_id',
    detail: 'this request id was already admitted with another payload'
  }
  await act(async () => {
    expect(await result.current('look at this', { submission_id: 'restored-1' })).toBe(true)
  })

  expect(requestGateway.mock.calls.filter(([method]) => method === 'prompt.submit')).toHaveLength(1)
  expect(state.messages.filter(message => message.role === 'user')).toHaveLength(0)
  expect(state.messages.some(message => message.error)).toBe(false)
  expect(state.busy).toBe(false)
  expect(window.localStorage.getItem('hermes.desktop.preparedSubmissions.v1') ?? '').not.toContain('restored-1')

  refusal = { reason: 'admission_conflict' }
  await act(async () => {
    expect(await result.current('another chat', { submission_id: 'restored-2' })).toBe(false)
  })

  expect(state.messages.some(message => message.error)).toBe(true)
})

// A local queue entry drains under its id. Its first drain can be admitted with the ack lost (the entry
// stays queued); an edit after that is a new message, so it drains under a new id and runs, instead
// of reading as that admitted original's retry and vanishing as "already delivered".
it('an edited queued entry drains as a new message, not as its admitted original', async () => {
  let state: ClientSessionState = createClientSessionState()
  const submitted: Record<string, unknown>[] = []

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'prompt.submit') {
      if (params?.submission_id === 'X-orig') {
        throw new JsonRpcGatewayError('admission_conflict', {
          code: 4001,
          data: { reason: 'admission_conflict', field: 'request_id' }
        })
      }

      submitted.push(params ?? {})

      return { submission_id: params?.submission_id, session_id: 'runtime-a', status: 'started' } as never
    }

    return {} as never
  })

  enqueueQueuedPrompt('stored-a', { id: 'X-orig', text: 'yes', attachments: [] })
  expect(updateQueuedPrompt('stored-a', 'X-orig', { text: 'yes, and also run the tests', attachments: [] })).toBe(true)
  const [entry] = getQueuedPrompts('stored-a')

  const deps = submitDeps(requestGateway as GatewayRequest, updater => (state = updater(state)))
  const { result } = renderHook(() => useSubmitPrompt(deps as never))
  await act(async () => {
    expect(
      await result.current(entry!.text, { fromQueue: true, submission_id: queuedSubmissionId(entry!), storedSessionId: 'stored-a' })
    ).toBe(true)
  })

  expect(submitted).toHaveLength(1)
  expect(submitted[0]).toMatchObject({ text: 'yes, and also run the tests' })
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
