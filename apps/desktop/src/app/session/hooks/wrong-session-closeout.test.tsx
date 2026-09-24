// @vitest-environment jsdom
/**
 * Integration invariant for #86106 / #86359: a background chat's delayed
 * stored-id rotation must not move the user off the surface they are on.
 *
 * The unit test in `use-session-state-cache.test.tsx` pins the PRODUCER (no
 * rotation event is published). This one wires the real producer to the real
 * CONSUMER (`useSessionActions`' route-follow effect) so the user-visible half
 * is pinned too: no navigate(), no selection move, no focus move.
 *
 *   A is the primary (route + selection + active runtime) → the user focuses
 *   tile C → A auto-compresses and rotates its stored id
 */
import { useStore } from '@nanostores/react'
import { act, cleanup, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { sessionRoute } from '@/app/routes'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { textPart } from '@/lib/chat-messages'
import { requestGatewayForAgent, requestGatewayForProfile } from '@/store/gateway'
import {
  $activeSessionId,
  $activeSessionStoredIdRotation,
  $selectedStoredSessionId,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { $focusedStoredSessionId, $sessionTiles, clearAllSessionStates } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { useSessionActions } from './use-session-actions'
import { useSessionStateCache } from './use-session-state-cache'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn(),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

let routedStoredId: string | null = 'stored-A'
const navigate = vi.fn()

let handle: { cache: ReturnType<typeof useSessionStateCache> }

function Harness() {
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const requestGateway = createSessionRpcDispatcher({
    ...cache,
    ambientRequest: async () => {
      throw new Error('unexpected ambient request')
    }
  })

  useSessionActions({
    activeSessionId,
    ...cache,
    busyRef,
    creatingSessionRef,
    getRouteToken: () => `${routedStoredId ? sessionRoute(routedStoredId) : '/'}::`,
    getRoutedStoredSessionId: () => routedStoredId,
    navigate: navigate as never,
    requestGateway,
    selectedStoredSessionId
  })

  handle = { cache }

  return null
}

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  setActiveSessionStoredIdRotation(null)
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  $sessionTiles.set([])
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
  setBusy(false)
  setAwaitingResponse(false)
  setMessages([])
  window.history.pushState({}, '', '/')
  window.localStorage.clear()
  vi.clearAllMocks()
})

it('a background chat’s delayed stored-id rotation never moves the user off the tile they are on', async () => {
  setSessions(
    ['A', 'C'].map(id => ({
      id: `stored-${id}`,
      connection_id: `connection-${id}`,
      profile: 'default',
      source: 'desktop',
      message_count: 1
    })) as SessionInfo[]
  )
  // A is the primary: selection, HashRouter route and active runtime all name it.
  routedStoredId = 'stored-A'
  window.history.pushState({}, '', '/#/stored-A')
  setSelectedStoredSessionId('stored-A')
  setActiveSessionId('rt-A')
  render(<Harness />)
  act(() => {
    for (const id of ['A', 'C']) {
      handle.cache.updateSessionState(
        `rt-${id}`,
        state => ({
          ...state,
          messages: [{ id: `history-${id}`, role: 'assistant', parts: [textPart(`history ${id}`)] }]
        }),
        `stored-${id}`
      )
    }
  })

  // The user opens C as a tile and is working there when the rotation lands.
  act(() => {
    $sessionTiles.set([{ runtimeId: 'rt-C', storedSessionId: 'stored-C' }])
    $layoutTree.set(group(['workspace', 'session-tile:stored-C'], { active: 'session-tile:stored-C', id: 'grp-main' }))
    noteActiveTreeGroup('grp-main')
  })
  expect($focusedStoredSessionId.get()).toBe('stored-C')

  // A's auto-compression rotates its stored id in the background (real
  // producer in useSessionStateCache, real consumer in useSessionActions).
  act(() => {
    handle.cache.updateSessionState('rt-A', state => state, 'stored-A-next')
  })
  await act(async () => undefined)

  expect(navigate).not.toHaveBeenCalled()
  expect($activeSessionStoredIdRotation.get()).toBeNull()
  expect($focusedStoredSessionId.get()).toBe('stored-C')
  expect($selectedStoredSessionId.get()).toBe('stored-A')
  expect($activeSessionId.get()).toBe('rt-A')
  // The rotation still re-keys A's own binding; only the foreground move is
  // suppressed.
  expect(handle.cache.runtimeIdByStoredSessionIdRef.current.get('stored-A-next')).toBe('rt-A')
  expect(handle.cache.runtimeIdByStoredSessionIdRef.current.has('stored-A')).toBe(false)
  expect(requestGatewayForAgent).not.toHaveBeenCalled()
  expect(requestGatewayForProfile).not.toHaveBeenCalled()
})
