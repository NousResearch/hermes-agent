import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { group } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $changeEventsAvailable, $sessionsChangeTick } from '@/store/live-sync'
import { $activeSessionId, $unreadFinishedSessionIds, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $sessionStates, clearAllSessionStates, reconcileBusyStatesOnReconnect } from '@/store/session-states'

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
  rehydrateLiveSessionStatuses(working('rt-primary'))
  reconcileBusyStatesOnReconnect()
  expect($unreadFinishedSessionIds.get()).toEqual([])
  focus('focus-b', 'other-connection', 'default')

  if (fails) {transport.request.mockRejectedValue(new Error('method not found'))}
  else {transport.request.mockResolvedValue(working('rt-other'))}

  mount()
  await act(async () => undefined)
  expect($unreadFinishedSessionIds.get()).toEqual([])
  rehydrateLiveSessionStatuses({ sessions: [] })
  expect($unreadFinishedSessionIds.get()).toContain('shared-stored-id')
})
