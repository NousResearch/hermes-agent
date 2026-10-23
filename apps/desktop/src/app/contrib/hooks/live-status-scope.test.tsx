import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { group } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $changeEventsAvailable, $sessionsChangeTick } from '@/store/live-sync'
import { $activeSessionId, $unreadFinishedSessionIds, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { pendingReconnectStatusSources } from '@/store/session-reconnect-settles'
import { $sessionStates, clearAllSessionStates, confirmReconnectSettlesExcept, reconcileBusyStatesOnReconnect } from '@/store/session-states'

import { rehydrateLiveSessionStatuses, resetLiveRuntimeTracking, useBackgroundSync } from './use-background-sync'

const transport = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: transport.request
}))
const noop = () => undefined
const selected = { current: null as null | string }

const dispatcher = createSessionRpcDispatcher({
  ambientRequest: transport.request,
  runtimeIdByStoredSessionIdRef: { current: new Map() },
  selectedStoredSessionIdRef: selected,
  sessionStateByRuntimeIdRef: { current: new Map() }
})

function mount() {
  return renderHook(() => useBackgroundSync({
    activeConnectionId: 'local', activeGatewayProfile: 'default',
    activeIsMessaging: false, activeSessionId: null, activeStoredSessionId: null,
    freshDraftReady: false, gatewayState: 'open',
    refreshActiveTranscript: noop, refreshCronJobs: noop, refreshCurrentModel: noop,
    refreshHermesConfig: noop, refreshMessagingSessions: noop, refreshSessions: noop,
    requestGateway: dispatcher, captureLiveStatusRequest: dispatcher.captureLiveStatusRequest,
    updateSessionState: vi.fn() as never
  }))
}

it('confirms a parked primary completion while another profile stays focused', async () => {
  rehydrateLiveSessionStatuses(working('rt-primary'), Date.now(), { connectionId: 'local', profile: 'default' })
  reconcileBusyStatesOnReconnect(undefined, { connectionId: 'local', profile: 'default' })
  focus('focus-b', 'local', 'probe-x')
  transport.request.mockResolvedValue({ sessions: [] })
  mount()
  await act(async () => undefined)
  expect(transport.request.mock.calls.map(call => call.slice(0, 2))).toContainEqual(['local', 'default'])
  expect($unreadFinishedSessionIds.get()).toContain('shared-stored-id')
})

const working = (id: string) => ({ sessions: [{ id, session_key: 'shared-stored-id', status: 'working' as const }] })

function focus(id: string, connectionId: string, profile: string, targetProfile?: string) {
  setSessionOwnerHint(id, { connectionId, profile, ...(targetProfile ? { targetProfile } : {}) })
  $layoutTree.set(group([`session-tile:${id}`], { active: `session-tile:${id}`, id: 'test-group' }))
  $activeTreeGroup.set('test-group')
}

beforeEach(() => {
  vi.useFakeTimers()
  clearAllSessionStates()
  resetLiveRuntimeTracking()
  transport.request.mockReset()
  _resetSessionOwnerHintsForTests({ storage: true })
  $unreadFinishedSessionIds.set([])
  $changeEventsAvailable.set(true)
  $sessionsChangeTick.set(0)
})
afterEach(() => {
  cleanup()
  clearAllSessionStates()
  resetLiveRuntimeTracking()
  $layoutTree.set(null)
  $activeTreeGroup.set(null)
  $activeSessionId.set(null)
  vi.clearAllTimers()
  vi.useRealTimers()
})

it.each([
  ['local', 'probe-b'],
  ['other-connection', 'probe-a']
])('does not reap a different source with the same stored id (%s/%s)', async (connection, profile) => {
  let bLive = true
  transport.request.mockImplementation(async (_connection, requestedProfile) =>
    requestedProfile === 'probe-a' && _connection === 'local' ? working('rt-a') : bLive ? working('rt-b') : { sessions: [] })
  focus('focus-a', 'local', 'probe-a')
  mount()
  await act(async () => undefined)
  expect($sessionStates.get()['rt-a']?.busy).toBe(true)
  focus('focus-b', connection, profile)
  await act(async () => { $sessionsChangeTick.set(1) })
  expect($sessionStates.get()['rt-a']?.busy).toBe(true)
  expect($sessionStates.get()['rt-b']?.busy).toBe(true)
  bLive = false
  await act(async () => { $sessionsChangeTick.set(2) })
  expect($sessionStates.get()['rt-b']?.busy).not.toBe(true)
  expect($sessionStates.get()['rt-a']?.busy).toBe(true)
})

it('keeps an in-flight A response bound to A after tile focus moves to B', async () => {
  let release!: (value: ReturnType<typeof working>) => void
  transport.request.mockResolvedValueOnce(working('rt-a'))
    .mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    .mockResolvedValue({ sessions: [] })
  focus('focus-a', 'local', 'probe-a')
  mount()
  await act(async () => undefined)
  await act(async () => { $sessionsChangeTick.set(1) })
  focus('focus-b', 'local', 'probe-b')
  await act(async () => { $sessionsChangeTick.set(2); release(working('rt-a')) })
  expect(transport.request.mock.calls.map(call => call.slice(0, 2))).toEqual([
    ['local', 'probe-a'], ['local', 'probe-a'], ['local', 'probe-b']
  ])
  expect($sessionStates.get()['rt-a']?.busy).toBe(true)
})

it('pins a remote target-profile override for both dispatch and absence tracking', async () => {
  transport.request.mockImplementation(async (_connection, _profile, _method, params) =>
    params.profile === 'target-a' ? working('rt-a') : { sessions: [] })
  focus('focus-a', 'remote', 'alias', 'target-a')
  mount()
  await act(async () => undefined)
  expect(transport.request).toHaveBeenLastCalledWith('remote', 'alias', 'session.active_list', { profile: 'target-a' })
  focus('focus-b', 'remote', 'alias', 'target-b')
  await act(async () => { $sessionsChangeTick.set(1) })
  expect(transport.request).toHaveBeenLastCalledWith('remote', 'alias', 'session.active_list', { profile: 'target-b' })
  expect($sessionStates.get()['rt-a']?.busy).toBe(true)
})

it.each([false, true])('only confirms parked reconnect completions from the captured scope (RPC fails: %s)', async fails => {
  const source = { connectionId: 'local', profile: 'default' }
  rehydrateLiveSessionStatuses(working('rt-primary'), Date.now(), source)
  reconcileBusyStatesOnReconnect(undefined, source)
  expect($unreadFinishedSessionIds.get()).toEqual([])
  focus('focus-b', 'other-connection', 'default')
  let finishPrimary!: (response: { sessions: [] }) => void
  transport.request.mockImplementation((connection: string) => {
    if (connection === 'local') { return new Promise(resolve => { finishPrimary = resolve }) }

    return fails ? Promise.reject(new Error('method not found')) : Promise.resolve(working('rt-other'))
  })

  mount()
  await act(async () => undefined)
  expect($unreadFinishedSessionIds.get()).toEqual([])
  await act(async () => { finishPrimary({ sessions: [] }) })
  expect($unreadFinishedSessionIds.get()).toContain('shared-stored-id')
})

it.each(['rt-old', 'rt-new'])('retires the old parked runtime when the same source reports %s still working', async runtimeId => {
  const source = { connectionId: 'local', profile: 'default' }
  rehydrateLiveSessionStatuses(working('rt-old'), Date.now(), source)
  reconcileBusyStatesOnReconnect(undefined, source)
  focus('focus-b', 'local', 'probe-x')
  transport.request.mockImplementation(async (_connection, profile) => profile === 'default' ? working(runtimeId) : { sessions: [] })
  mount()
  await act(async () => undefined)
  expect(transport.request.mock.calls.map(call => call.slice(0, 2))).toContainEqual(['local', 'default'])
  expect($sessionStates.get()[runtimeId]?.busy).toBe(true)
  expect($unreadFinishedSessionIds.get()).toEqual([])
  expect(pendingReconnectStatusSources()).toEqual([])
  // A later fallback must not resurrect the old runtime's parked completion.
  confirmReconnectSettlesExcept(new Set(), 'conn:local::default')
  expect($unreadFinishedSessionIds.get()).toEqual([])
})

it.each(['method not found', 'request timed out'])('falls back for a pending source only after its own %s', async message => {
  const source = { connectionId: 'local', profile: 'default' }
  rehydrateLiveSessionStatuses(working('rt-primary'), Date.now(), source)
  reconcileBusyStatesOnReconnect(undefined, source)
  focus('focus-b', 'local', 'probe-x')
  transport.request.mockImplementation(async (_connection, profile) => {
    if (profile === 'default') { throw new Error(message) }

    return working('rt-other')
  })
  mount()
  await act(async () => undefined)
  expect($unreadFinishedSessionIds.get()).toContain('shared-stored-id')
  expect($sessionStates.get()['rt-other']?.busy).toBe(true)
  expect(pendingReconnectStatusSources()).toEqual([])
})

it('deduplicates the focused source when it is also awaiting reconnect confirmation', async () => {
  const source = { connectionId: 'local', profile: 'default' }
  rehydrateLiveSessionStatuses(working('rt-primary'), Date.now(), source)
  reconcileBusyStatesOnReconnect(undefined, source)
  focus('focus-a', 'local', 'default')
  transport.request.mockResolvedValue({ sessions: [] })
  mount()
  await act(async () => undefined)
  expect(transport.request).toHaveBeenCalledTimes(1)
  expect($unreadFinishedSessionIds.get()).toContain('shared-stored-id')
})
