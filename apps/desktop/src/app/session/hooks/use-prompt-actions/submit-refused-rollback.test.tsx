import { JsonRpcGatewayError } from '@hermes/shared'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { $shellChoice } from '@/store/shell-variant'

import { useSubmitPrompt } from './submit'
import type { GatewayRequest } from './utils'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

type ClientSessionState = ReturnType<typeof createClientSessionState>

beforeEach(() => {
  window.localStorage.clear()
  $sessions.set([])
  $sessionStates.set({})
  $connection.set(null)
  $activeGatewayProfile.set('default')
  clearNotifications()
})

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
  vi.restoreAllMocks()
})

function refusedSend() {
  let state: ClientSessionState = createClientSessionState()

  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'prompt.submit') {
      throw new JsonRpcGatewayError('Invalid params', { code: 4000 })
    }

    return {} as never
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

  return { result, state: () => state }
}

const failureToasts = () => $notifications.get().filter(item => item.title === en.desktop.promptFailed)

// The composer stashes a typed send and loads it back on refusal (use-composer-submit), so a refused
// composer send must leave exactly one copy of the words and one error surface.
it('redesign: a refused composer send rolls back whole, leaving the toast as its one error surface', async () => {
  $shellChoice.set(true)
  const { result, state } = refusedSend()

  await act(async () => {
    expect(await result.current('reply with pong', { composerScope: 'stored-a', submission_id: 's-1' })).toBe(false)
  })

  expect(state().messages.filter(message => message.role === 'user')).toHaveLength(0)
  expect(state().messages.some(message => message.error)).toBe(false)
  expect(state().busy).toBe(false)
  expect(failureToasts()).toHaveLength(1)
})

it('redesign: a send nothing gives back keeps its bubble and the in-thread error card', async () => {
  $shellChoice.set(true)
  const { result, state } = refusedSend()

  // No composerScope: a queue drain or an external request, whose words the composer does not reload.
  await act(async () => {
    expect(await result.current('reply with pong', { submission_id: 's-2' })).toBe(false)
  })

  expect(state().messages.filter(message => message.role === 'user')).toHaveLength(1)
  expect(state().messages.some(message => message.error)).toBe(true)
})

it('classic: a refused composer send keeps the bubble with its error card, as on main', async () => {
  $shellChoice.set(false)
  const { result, state } = refusedSend()

  await act(async () => {
    expect(await result.current('reply with pong', { composerScope: 'stored-a', submission_id: 's-3' })).toBe(false)
  })

  expect(state().messages.filter(message => message.role === 'user')).toHaveLength(1)
  expect(state().messages.some(message => message.error)).toBe(true)
})
