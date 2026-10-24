import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { group } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $changeEventsAvailable, $sessionsChangeTick } from '@/store/live-sync'
import {
  $activeSessionId,
  $unreadFinishedSessionIds,
  _resetSessionOwnerHintsForTests,
  setSessionOwnerHint
} from '@/store/session'
import {
  $sessionStates,
  clearAllSessionStates,
  confirmReconnectSettlesExcept,
  reconcileBusyStatesOnReconnect
} from '@/store/session-states'

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
  return renderHook(() =>
    useBackgroundSync({
      activeConnectionId: 'local',
      activeGatewayProfile: 'default',
      activeIsMessaging: false,
      activeSessionId: null,
      activeStoredSessionId: null,
      freshDraftReady: false,
      gatewayState: 'open',
      refreshActiveTranscript: noop,
      refreshCronJobs: noop,
      refreshCurrentModel: noop,
      refreshHermesConfig: noop,
      refreshMessagingSessions: noop,
      refreshSessions: noop,
      requestGateway: dispatcher,
      captureLiveStatusRequest: dispatcher.captureLiveStatusRequest,
      updateSessionState: vi.fn() as never
    })
  )
}

const working = (id: string) => ({ sessions: [{ id, session_key: 'shared-stored-id', status: 'working' as const }] })

function focus(id: string, connectionId: string, profile: string, targetProfile?: string) {
  setSessionOwnerHint(id, { connectionId, profile, ...(targetProfile ? { targetProfile } : {}) })
  $layoutTree.set(group([`session-tile:${id}`], { active: `session-tile:${id}`, id: 'test-group' }))
  $activeTreeGroup.set('test-group')
}

function resetFixture() {
  vi.useFakeTimers()
  clearAllSessionStates()
  resetLiveRuntimeTracking()
  transport.request.mockReset()
  _resetSessionOwnerHintsForTests({ storage: true })
  $unreadFinishedSessionIds.set([])
  $changeEventsAvailable.set(true)
  $sessionsChangeTick.set(0)
}

function disposeFixture() {
  cleanup()
  clearAllSessionStates()
  resetLiveRuntimeTracking()
  $layoutTree.set(null)
  $activeTreeGroup.set(null)
  $activeSessionId.set(null)
  vi.clearAllTimers()
  vi.useRealTimers()
}

beforeEach(resetFixture)
afterEach(disposeFixture)

interface Scenario {
  name: string
  run: () => Promise<void>
}

async function checkScenarios(scenarios: Scenario[]) {
  const failures: string[] = []

  for (const scenario of scenarios) {
    resetFixture()

    try {
      await scenario.run()
    } catch (error) {
      failures.push(`${scenario.name}: ${error instanceof Error ? error.stack : String(error)}`)
    } finally {
      disposeFixture()
    }
  }

  expect(failures, 'All invariant table rows must hold').toEqual([])
}

async function expectOnlyFocusedPoll(connection: string, profile: string) {
  transport.request.mockClear()
  await act(async () => {
    $sessionsChangeTick.set($sessionsChangeTick.get() + 1)
  })
  expect(transport.request.mock.calls.map(call => call.slice(0, 2))).toEqual([[connection, profile]])
}

const responseScenarios: Scenario[] = [
  ...[
    ['local', 'probe-b'],
    ['other-connection', 'probe-a']
  ].map(([connection, profile]) => ({
    name: `focus to ${connection}/${profile}: equal stored IDs stay isolated`,
    run: async () => {
      let bLive = true
      transport.request.mockImplementation(async (_connection, requestedProfile) =>
        requestedProfile === 'probe-a' && _connection === 'local'
          ? working('rt-a')
          : bLive
            ? working('rt-b')
            : { sessions: [] }
      )
      focus('focus-a', 'local', 'probe-a')
      mount()
      await act(async () => undefined)
      expect($sessionStates.get()['rt-a']?.busy).toBe(true)
      focus('focus-b', connection, profile)
      await act(async () => {
        $sessionsChangeTick.set(1)
      })
      expect($sessionStates.get()['rt-a']?.busy).toBe(true)
      expect($sessionStates.get()['rt-b']?.busy).toBe(true)
      bLive = false
      await act(async () => {
        $sessionsChangeTick.set(2)
      })
      expect($sessionStates.get()['rt-b']?.busy).not.toBe(true)
      expect($sessionStates.get()['rt-a']?.busy).toBe(true)
    }
  })),
  {
    name: 'in-flight response keeps its captured source after focus moves',
    run: async () => {
      let release!: (value: ReturnType<typeof working>) => void
      transport.request
        .mockResolvedValueOnce(working('rt-a'))
        .mockImplementationOnce(
          () =>
            new Promise(resolve => {
              release = resolve
            })
        )
        .mockResolvedValue({ sessions: [] })
      focus('focus-a', 'local', 'probe-a')
      mount()
      await act(async () => undefined)
      await act(async () => {
        $sessionsChangeTick.set(1)
      })
      focus('focus-b', 'local', 'probe-b')
      await act(async () => {
        $sessionsChangeTick.set(2)
        release(working('rt-a'))
      })
      expect(transport.request.mock.calls.map(call => call.slice(0, 2))).toEqual([
        ['local', 'probe-a'],
        ['local', 'probe-a'],
        ['local', 'probe-b']
      ])
      expect($sessionStates.get()['rt-a']?.busy).toBe(true)
    }
  },
  {
    name: 'remote target override binds dispatch and absence tracking',
    run: async () => {
      transport.request.mockImplementation(async (_connection, _profile, _method, params) =>
        params.profile === 'target-a' ? working('rt-a') : { sessions: [] }
      )
      focus('focus-a', 'remote', 'alias', 'target-a')
      mount()
      await act(async () => undefined)
      expect(transport.request).toHaveBeenLastCalledWith('remote', 'alias', 'session.active_list', {
        profile: 'target-a'
      })
      focus('focus-b', 'remote', 'alias', 'target-b')
      await act(async () => {
        $sessionsChangeTick.set(1)
      })
      expect(transport.request).toHaveBeenLastCalledWith('remote', 'alias', 'session.active_list', {
        profile: 'target-b'
      })
      expect($sessionStates.get()['rt-a']?.busy).toBe(true)
    }
  }
]

const reconnectScenarios: Scenario[] = [
  {
    name: 'confirm primary while another profile remains focused',
    run: async () => {
      rehydrateLiveSessionStatuses(working('rt-primary'), Date.now(), { connectionId: 'local', profile: 'default' })
      reconcileBusyStatesOnReconnect(undefined, { connectionId: 'local', profile: 'default' })
      focus('focus-b', 'local', 'probe-x')
      transport.request.mockResolvedValue({ sessions: [] })
      mount()
      await act(async () => undefined)
      expect(transport.request.mock.calls.map(call => call.slice(0, 2))).toContainEqual(['local', 'default'])
      expect($unreadFinishedSessionIds.get()).toContain('shared-stored-id')
    }
  },
  ...[false, true].map(fails => ({
    name: `other connection fails=${fails}: cannot confirm primary`,
    run: async () => {
      const source = { connectionId: 'local', profile: 'default' }
      rehydrateLiveSessionStatuses(working('rt-primary'), Date.now(), source)
      reconcileBusyStatesOnReconnect(undefined, source)
      expect($unreadFinishedSessionIds.get()).toEqual([])
      focus('focus-b', 'other-connection', 'default')
      let finishPrimary!: (response: { sessions: [] }) => void
      transport.request.mockImplementation((connection: string) => {
        if (connection === 'local') {
          return new Promise(resolve => {
            finishPrimary = resolve
          })
        }

        return fails ? Promise.reject(new Error('method not found')) : Promise.resolve(working('rt-other'))
      })

      mount()
      await act(async () => undefined)
      expect($unreadFinishedSessionIds.get()).toEqual([])
      expect(transport.request.mock.calls.map(call => call.slice(0, 2))).toContainEqual(['local', 'default'])
      await act(async () => {
        finishPrimary({ sessions: [] })
      })
      expect($unreadFinishedSessionIds.get()).toContain('shared-stored-id')
    }
  })),
  ...['rt-old', 'rt-new'].map(runtimeId => ({
    name: `same-source ${runtimeId} working retires old park without unread`,
    run: async () => {
      const source = { connectionId: 'local', profile: 'default' }
      rehydrateLiveSessionStatuses(working('rt-old'), Date.now(), source)
      reconcileBusyStatesOnReconnect(undefined, source)
      focus('focus-b', 'local', 'probe-x')
      transport.request.mockImplementation(async (_connection, profile) =>
        profile === 'default' ? working(runtimeId) : { sessions: [] }
      )
      mount()
      await act(async () => undefined)
      expect(transport.request.mock.calls.map(call => call.slice(0, 2))).toContainEqual(['local', 'default'])
      expect($sessionStates.get()[runtimeId]?.busy).toBe(true)
      expect($unreadFinishedSessionIds.get()).toEqual([])
      await expectOnlyFocusedPoll('local', 'probe-x')
      // A later fallback must not resurrect the old runtime's parked completion.
      confirmReconnectSettlesExcept(new Set(), 'conn:local::default')
      expect($unreadFinishedSessionIds.get()).toEqual([])
    }
  })),
  ...['method not found', 'request timed out'].map(message => ({
    name: `pending source ${message}: fallback only within source`,
    run: async () => {
      const source = { connectionId: 'local', profile: 'default' }
      rehydrateLiveSessionStatuses(working('rt-primary'), Date.now(), source)
      reconcileBusyStatesOnReconnect(undefined, source)
      focus('focus-b', 'local', 'probe-x')
      transport.request.mockImplementation(async (_connection, profile) => {
        if (profile === 'default') {
          throw new Error(message)
        }

        return working('rt-other')
      })
      mount()
      await act(async () => undefined)
      expect($unreadFinishedSessionIds.get()).toContain('shared-stored-id')
      expect($sessionStates.get()['rt-other']?.busy).toBe(true)
      await expectOnlyFocusedPoll('local', 'probe-x')
    }
  })),
  {
    name: 'focused pending source dispatches exactly once',
    run: async () => {
      const source = { connectionId: 'local', profile: 'default' }
      rehydrateLiveSessionStatuses(working('rt-primary'), Date.now(), source)
      reconcileBusyStatesOnReconnect(undefined, source)
      focus('focus-a', 'local', 'default')
      transport.request.mockResolvedValue({ sessions: [] })
      mount()
      await act(async () => undefined)
      expect(transport.request).toHaveBeenCalledTimes(1)
      expect($unreadFinishedSessionIds.get()).toContain('shared-stored-id')
    }
  }
]

it('reconciles each response only under its request-time source', async () => {
  await checkScenarios(responseScenarios)
})

it('confirms reconnect parks only within their source regardless of focus', async () => {
  await checkScenarios(reconnectScenarios)
})
