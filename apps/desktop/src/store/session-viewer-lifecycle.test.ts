import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { useSessionStateCache } from '@/app/session/hooks/use-session-state-cache'
import { createClientSessionState } from '@/lib/chat-runtime'

import { $activeSessionId, $selectedStoredSessionId } from './session'
import { $sessionStates, $sessionTiles, clearAllSessionStates, closeSessionTile, dropSessionState, publishSessionState, recordSessionEventScope } from './session-states'
import { deferCloseoutViewer, noteSessionViewerIdle } from './session-viewer-lifecycle'

const { request, gatewayRouteState, notifyError } = vi.hoisted(() => ({ request: vi.fn(), gatewayRouteState: vi.fn(), notifyError: vi.fn() }))
vi.mock('./gateway', async importOriginal => ({ ...await importOriginal<object>(), gatewayRouteState }))
vi.mock('./notifications', async importOriginal => ({ ...await importOriginal<object>(), notifyError }))

let gateway: { request: typeof request; connectionState: string }
const ownerRoute = { connectionId: 'cloud-a', profile: 'desktop-bot', targetProfile: 'bot-a' }
const state = (busy = false) => ({ ...createClientSessionState('stored'), busy })

const flush = async () => { await Promise.resolve(); await Promise.resolve() }

afterEach(cleanup)

beforeEach(() => {
  clearAllSessionStates()
  $sessionTiles.set([])
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  request.mockReset().mockResolvedValue({ detached: true })
  notifyError.mockReset()
  gateway = { request, connectionState: 'open' }
  gatewayRouteState.mockReset().mockReturnValue({ gateway, open: true, key: 'cloud-a:desktop-bot' })
})

it('closing an idle tab detaches only its captured owner on the existing socket', async () => {
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute }])
  publishSessionState('runtime', state())
  closeSessionTile('stored')
  expect($sessionStates.get().runtime).toBeUndefined()
  await flush()
  expect(gatewayRouteState).toHaveBeenCalledWith('cloud-a', 'desktop-bot')
  expect(request).toHaveBeenCalledExactlyOnceWith('session.detach', { session_id: 'runtime', profile: 'bot-a' })
})

it('a closed busy tab keeps streaming and detaches once only after settlement', async () => {
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute }])
  publishSessionState('runtime', state(true))
  closeSessionTile('stored')
  await flush()
  expect(request).not.toHaveBeenCalled()
  publishSessionState('runtime', state())
  await flush()
  publishSessionState('runtime', state())
  await flush()
  expect(request).toHaveBeenCalledTimes(1)
  expect($sessionStates.get().runtime?.busy).toBe(false)
})

it.each(['reopened', 'submit', 'main'] as const)('cancels a pending detach when %s takes the runtime', async kind => {
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute }])
  publishSessionState('runtime', state())
  closeSessionTile('stored')

  if (kind === 'reopened') { $sessionTiles.set([{ storedSessionId: 'stored', ownerRoute }]) }

  if (kind === 'submit') { publishSessionState('runtime', state(true)) }

  if (kind === 'main') { $selectedStoredSessionId.set('stored') }
  await flush()
  expect(request).not.toHaveBeenCalled()
})

it('never moves cleanup onto a replacement socket or guesses an absent owner', async () => {
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute }])
  publishSessionState('runtime', state())
  closeSessionTile('stored')
  gatewayRouteState.mockReturnValue({ gateway: { request: vi.fn() }, open: true })
  await flush()
  expect(request).not.toHaveBeenCalled()
  $sessionTiles.set([{ storedSessionId: 'unknown', runtimeId: 'unknown-runtime' }])
  publishSessionState('unknown-runtime', createClientSessionState('unknown'))
  closeSessionTile('unknown')
  await flush()
  expect(request).not.toHaveBeenCalled()
})

it('releases an abandoned idle runtime when the same stored tab binds a replacement', async () => {
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute }])
  publishSessionState('runtime', state())
  dropSessionState('runtime')
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'replacement', ownerRoute }])
  await flush()
  expect(request).toHaveBeenCalledExactlyOnceWith('session.detach', { session_id: 'runtime', profile: 'bot-a' })
})

it('keeps a refused detach safe and reports unsupported cleanup once without retries or close/reset', async () => {
  request.mockResolvedValueOnce({ detached: false }).mockRejectedValue(new Error('Method not found: session.detach'))

  for (const runtimeId of ['busy-on-server', 'unsupported', 'another']) {
    $sessionTiles.set([{ storedSessionId: 'stored', runtimeId, ownerRoute }])
    publishSessionState(runtimeId, state())
    closeSessionTile('stored')
    await flush()
  }

  expect(request.mock.calls.map(call => call[0])).toEqual(['session.detach', 'session.detach', 'session.detach'])
  expect(notifyError).toHaveBeenCalledTimes(1)
})

it('retries a refused completion detach once on fresh final-idle evidence, never on repeated paints', async () => {
  request.mockResolvedValueOnce({ detached: false }).mockResolvedValue({ detached: true })
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute }])
  publishSessionState('runtime', state(true))
  closeSessionTile('stored')
  publishSessionState('runtime', state())
  await flush()
  expect(request).toHaveBeenCalledTimes(1)
  publishSessionState('runtime', state())
  await flush()
  expect(request).toHaveBeenCalledTimes(1)
  noteSessionViewerIdle('runtime')
  await flush()
  expect(request).toHaveBeenCalledTimes(2)
  noteSessionViewerIdle('runtime')
  await flush()
  expect(request).toHaveBeenCalledTimes(2)
})

it('releases the last main-view reference while its transcript remains in the warm cache', async () => {
  $activeSessionId.set('runtime')
  $selectedStoredSessionId.set('stored')
  recordSessionEventScope({ session_id: 'runtime', connectionId: 'cloud-a', profile: 'bot-a' })

  const { result, rerender } = renderHook(({ active, stored }) => useSessionStateCache({
    activeSessionId: active,
    selectedStoredSessionId: stored,
    busyRef: { current: false },
    setMessages: vi.fn(), setBusy: vi.fn(), setAwaitingResponse: vi.fn()
  }), { initialProps: { active: 'runtime', stored: 'stored' } })

  act(() => { result.current.updateSessionState('runtime', () => state(), 'stored') })
  $activeSessionId.set('new-runtime')
  $selectedStoredSessionId.set('new-stored')
  rerender({ active: 'new-runtime', stored: 'new-stored' })
  await act(flush)
  expect(result.current.sessionStateByRuntimeIdRef.current.has('runtime')).toBe(true)
  expect(request).toHaveBeenCalledExactlyOnceWith('session.detach', { session_id: 'runtime', profile: 'bot-a' })
})

it('a queued idle callback cannot detach a replacement viewer after a connection reset', async () => {
  request.mockResolvedValue({ detached: false })
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute }])
  publishSessionState('runtime', state(true))
  closeSessionTile('stored')
  publishSessionState('runtime', state())
  await flush()
  expect(request).toHaveBeenCalledTimes(1)
  noteSessionViewerIdle('runtime')
  clearAllSessionStates()
  const replacementRequest = vi.fn(async () => ({ detached: true }))
  gatewayRouteState.mockReturnValue({ gateway: { request: replacementRequest }, open: true, key: 'replacement' })
  recordSessionEventScope({ session_id: 'runtime', connectionId: 'other-cloud', profile: 'other-bot' })
  publishSessionState('runtime', state())
  await flush()
  expect(replacementRequest).not.toHaveBeenCalled()
  expect(request).toHaveBeenCalledTimes(1)
})

it('a durable closeout can release the UI before physical settlement, then detaches only at authoritative idle', async () => {
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute }])
  publishSessionState('runtime', state(true))
  deferCloseoutViewer('runtime', 0)
  closeSessionTile('stored')
  await flush()
  expect(request).not.toHaveBeenCalled()
  noteSessionViewerIdle('runtime')
  publishSessionState('runtime', state())
  await flush()
  expect(request).toHaveBeenCalledExactlyOnceWith('session.detach', { session_id: 'runtime', profile: 'bot-a' })
})

it('does not lose a final idle signal that arrived before the closeout UI was applied', async () => {
  $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute }])
  publishSessionState('runtime', state(true))
  noteSessionViewerIdle('runtime')
  publishSessionState('runtime', state())
  deferCloseoutViewer('runtime', 0)
  closeSessionTile('stored')
  await flush()
  expect(request).toHaveBeenCalledExactlyOnceWith('session.detach', { session_id: 'runtime', profile: 'bot-a' })
})
