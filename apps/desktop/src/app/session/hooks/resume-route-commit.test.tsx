import { cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { ensureGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  setActiveSessionId,
  setConnection,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

import { sessionRoute } from '../../routes'

import { useSessionActions } from './use-session-actions'
import { useSessionStateCache } from './use-session-state-cache'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))
vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  setSessions([])
  setMessages([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  vi.restoreAllMocks()
})

it('binds the resumed session when the route commits onto it mid-resume', async () => {
  // Branch/create call navigate(child) and then resume(child): the route
  // token still names the parent at entry and only flips to the child while
  // the resume is awaiting the gateway.
  let routeToken = `${sessionRoute('parent')}::`
  setConnection(null)
  setSessions([])
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: 'child' })
  vi.mocked(ensureGatewayProfile).mockImplementationOnce(async () => {
    routeToken = `${sessionRoute('child')}::`
  })

  const requestGateway = vi.fn().mockResolvedValue({
    info: {},
    message_count: 0,
    messages: [],
    resumed: 'child',
    session_id: 'child-runtime',
    session_key: 'child'
  })

  const { result } = renderHook(() => {
    const busyRef = useRef(false)

    const cache = useSessionStateCache({
      activeSessionId: null,
      busyRef,
      selectedStoredSessionId: null,
      setAwaitingResponse: vi.fn(),
      setBusy: vi.fn(),
      setMessages
    })

    return useSessionActions({
      ...cache,
      activeSessionId: null,
      busyRef,
      creatingSessionRef: useRef(false),
      getRouteToken: () => routeToken,
      getRoutedStoredSessionId: () => null,
      navigate: vi.fn(),
      requestGateway,
      routedSessionId: null,
      selectedStoredSessionId: null
    })
  })

  await result.current.resumeSession('child', true)

  expect($activeSessionId.get()).toBe('child-runtime')
})
