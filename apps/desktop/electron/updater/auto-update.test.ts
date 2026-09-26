import { describe, expect, it } from 'vitest'

import {
  AUTO_UPDATE_MAX_DEFER_MS,
  autoUpdateSessionScope,
  type AutoUpdateState,
  decideAutoUpdateClaim,
  DEFAULT_AUTO_UPDATE_STATE,
  gatewayRecordBusy,
  type LoginSessionDeps,
  parseAutoUpdateState,
  recordAutoUpdateOutcome,
  resolveLoginSessionKey,
  scanGatewayActivity
} from './auto-update'

const NOW = Date.parse('2026-09-26T09:00:00Z')
const enabled: AutoUpdateState = { ...DEFAULT_AUTO_UPDATE_STATE, enabled: true }

function claim(overrides: Partial<Parameters<typeof decideAutoUpdateClaim>[0]> = {}) {
  return decideAutoUpdateClaim({
    state: enabled,
    sessionKey: 'login-A',
    platform: 'darwin',
    gatewayActiveAgents: 0,
    desktopActiveTurns: 0,
    now: NOW,
    ...overrides
  })
}

describe('decideAutoUpdateClaim', () => {
  it('does nothing while the setting is off (the default)', () => {
    const { claim: c, nextState } = claim({ state: DEFAULT_AUTO_UPDATE_STATE })

    expect(c).toMatchObject({ action: 'skip', reason: 'disabled' })
    expect(nextState).toBe(DEFAULT_AUTO_UPDATE_STATE)
  })

  it('runs on the first launch of a login session and consumes the session up front', () => {
    const { claim: c, nextState } = claim()

    expect(c).toMatchObject({ action: 'run', reason: 'first-launch-after-login', sessionKey: 'login-A' })
    expect(nextState.claimedSessionKey).toBe('login-A')
  })

  it('never runs twice in one login session — relaunch after update, crash, or failure', () => {
    const first = claim().nextState
    const again = claim({ state: first })

    expect(again.claim).toMatchObject({ action: 'skip', reason: 'already-ran-this-session' })

    const failed = recordAutoUpdateOutcome(first, { sessionKey: 'login-A', outcome: 'failed', message: 'boom' }, NOW)

    expect(claim({ state: failed }).claim.action).toBe('skip')
  })

  it('runs again after the next login', () => {
    const afterFirst = claim().nextState

    expect(claim({ state: afterFirst, sessionKey: 'login-B' }).claim.action).toBe('run')
  })

  it('supports macOS and Linux only', () => {
    expect(claim({ platform: 'linux' }).claim.action).toBe('run')
    expect(claim({ platform: 'win32' }).claim).toMatchObject({ action: 'skip', reason: 'unsupported-platform' })
  })

  it('defers without consuming the session while a gateway or this Desktop is mid-turn', () => {
    const gateway = claim({ gatewayActiveAgents: 2 })

    expect(gateway.claim).toMatchObject({ action: 'defer', reason: 'busy', activeAgents: 2 })
    expect(gateway.nextState.claimedSessionKey).toBeUndefined()
    expect(gateway.nextState.deferral).toEqual({ sessionKey: 'login-A', since: NOW })

    expect(claim({ desktopActiveTurns: 1 }).claim.action).toBe('defer')
  })

  it('keeps the original deferral start across retries, then runs once the work drains', () => {
    const first = claim({ gatewayActiveAgents: 1 }).nextState
    const second = claim({ state: first, gatewayActiveAgents: 1, now: NOW + 60_000 })

    expect(second.nextState.deferral?.since).toBe(NOW)

    const drained = claim({ state: second.nextState, now: NOW + 120_000 })

    expect(drained.claim.action).toBe('run')
    expect(drained.nextState.deferral).toBeUndefined()
  })

  it('gives up on the session after the deferral budget instead of waiting forever', () => {
    const first = claim({ gatewayActiveAgents: 1 }).nextState
    const late = claim({ state: first, gatewayActiveAgents: 1, now: NOW + AUTO_UPDATE_MAX_DEFER_MS })

    expect(late.claim).toMatchObject({ action: 'skip', reason: 'deferred-too-long' })
    expect(late.nextState.claimedSessionKey).toBe('login-A')
    expect(late.nextState.lastAttempt?.outcome).toBe('deferred-timeout')
  })

  it('does not inherit a deferral window from a previous login', () => {
    const stale: AutoUpdateState = {
      ...enabled,
      deferral: { sessionKey: 'login-OLD', since: NOW - 10 * AUTO_UPDATE_MAX_DEFER_MS }
    }

    expect(claim({ state: stale, gatewayActiveAgents: 1 }).claim.action).toBe('defer')
  })
})

describe('parseAutoUpdateState', () => {
  it('defaults to off for missing or garbage files', () => {
    expect(parseAutoUpdateState(null)).toEqual(DEFAULT_AUTO_UPDATE_STATE)
    expect(parseAutoUpdateState('nope')).toEqual(DEFAULT_AUTO_UPDATE_STATE)
    expect(parseAutoUpdateState({ enabled: 'yes' }).enabled).toBe(false)
  })

  it('round-trips a valid state and drops unknown outcomes', () => {
    const state = recordAutoUpdateOutcome(
      claim().nextState,
      { sessionKey: 'login-A', outcome: 'handed-off', target: 'main' },
      NOW
    )

    expect(parseAutoUpdateState(JSON.parse(JSON.stringify(state)))).toEqual(state)
    expect(
      parseAutoUpdateState({ enabled: true, lastAttempt: { sessionKey: 'x', at: 1, outcome: 'exploded' } }).lastAttempt
    ).toBeUndefined()
  })
})

describe('gatewayRecordBusy', () => {
  const alive = () => true
  const fresh = new Date(NOW - 10_000).toISOString()

  it('counts active agents on a live, fresh gateway', () => {
    expect(
      gatewayRecordBusy({ gateway_state: 'running', pid: 42, updated_at: fresh, active_agents: 3 }, NOW, alive)
    ).toBe(3)
    expect(
      gatewayRecordBusy({ gateway_state: 'draining', pid: 42, updated_at: fresh, active_agents: 1 }, NOW, alive)
    ).toBe(1)
  })

  it('ignores idle, stopped, dead-pid, and stale-heartbeat records', () => {
    const base = { gateway_state: 'running', pid: 42, updated_at: fresh, active_agents: 2 }

    expect(gatewayRecordBusy({ ...base, active_agents: 0 }, NOW, alive)).toBe(0)
    expect(gatewayRecordBusy({ ...base, gateway_state: 'stopped' }, NOW, alive)).toBe(0)
    expect(gatewayRecordBusy(base, NOW, () => false)).toBe(0)
    expect(gatewayRecordBusy({ ...base, updated_at: new Date(NOW - 10 * 60_000).toISOString() }, NOW, alive)).toBe(0)
    expect(gatewayRecordBusy({ ...base, updated_at: 'garbage' }, NOW, alive)).toBe(0)
  })
})

describe('scanGatewayActivity', () => {
  it('sums busy gateways across the root home and every profile', () => {
    const fresh = new Date(NOW - 5_000).toISOString()

    const files: Record<string, string> = {
      '/h/gateway_state.json': JSON.stringify({
        gateway_state: 'running',
        pid: 1,
        updated_at: fresh,
        active_agents: 1
      }),
      '/h/profiles/work/gateway_state.json': JSON.stringify({
        gateway_state: 'running',
        pid: 2,
        updated_at: fresh,
        active_agents: 2
      }),
      '/h/profiles/idle/gateway_state.json': JSON.stringify({
        gateway_state: 'running',
        pid: 3,
        updated_at: fresh,
        active_agents: 0
      })
    }

    const activity = scanGatewayActivity({
      hermesHome: '/h',
      readFile: file => {
        if (!(file in files)) {
          throw new Error('ENOENT')
        }

        return files[file]
      },
      listDir: () => ['work', 'idle', 'empty'],
      isPidAlive: () => true,
      now: () => NOW
    })

    expect(activity).toEqual({ busy: true, activeAgents: 3, busyHomes: ['/h', '/h/profiles/work'] })
  })
})

describe('resolveLoginSessionKey', () => {
  function deps(overrides: Partial<LoginSessionDeps>): LoginSessionDeps {
    return {
      platform: 'darwin',
      uid: 501,
      env: {},
      run: () => {
        throw new Error('not stubbed')
      },
      readFile: () => {
        throw new Error('not stubbed')
      },
      uptimeSeconds: () => 1000,
      now: () => NOW,
      ...overrides
    }
  }

  it('macOS: keys on the user loginwindow pid + start time', () => {
    const run = (command: string, args: string[]) => {
      if (command === '/usr/bin/pgrep') {
        expect(args).toEqual(['-u', '501', '-x', 'loginwindow'])

        return '598\n'
      }

      if (command === '/bin/ps') {
        return 'Tue Sep 15 19:39:38   2026\n'
      }

      throw new Error(command)
    }

    expect(resolveLoginSessionKey(deps({ run }))).toBe('darwin:loginwindow:598:Tue Sep 15 19:39:38 2026')
  })

  it('macOS: falls back to boot time when loginwindow is unreadable', () => {
    const run = (command: string) => {
      if (command === '/usr/sbin/sysctl') {
        return '{ sec = 1789481317, usec = 785236 } Tue Sep 15 19:38:37 2026'
      }

      throw new Error(command)
    }

    expect(resolveLoginSessionKey(deps({ run }))).toBe('darwin:boot:1789481317')
  })

  it('Linux: logind session id scoped by kernel boot id', () => {
    const key = resolveLoginSessionKey(
      deps({ platform: 'linux', env: { XDG_SESSION_ID: '7' }, readFile: () => 'abc-123\n' })
    )

    expect(key).toBe('linux:abc-123:session:7')
  })

  it('Linux without logind: boot id alone (first launch after boot)', () => {
    expect(resolveLoginSessionKey(deps({ platform: 'linux', readFile: () => 'abc-123' }))).toBe('linux:abc-123')
  })

  it('is stable across launches in the same boot when nothing is readable', () => {
    const a = resolveLoginSessionKey(deps({ platform: 'freebsd', uptimeSeconds: () => 1000, now: () => NOW }))
    const b = resolveLoginSessionKey(deps({ platform: 'freebsd', uptimeSeconds: () => 1060, now: () => NOW + 60_000 }))

    expect(a).toBe(b)
  })
})

describe('autoUpdateSessionScope', () => {
  it('names login-scoped keys so the UI can promise "after you log in"', () => {
    expect(autoUpdateSessionScope('darwin:loginwindow:598:Tue Sep 15 19:39:38 2026')).toBe('login')
    expect(autoUpdateSessionScope('linux:abc-123:session:4')).toBe('login')
  })

  it('names boot-scoped fallbacks so the UI says "after this computer starts" instead', () => {
    expect(autoUpdateSessionScope('linux:abc-123')).toBe('boot')
    expect(autoUpdateSessionScope('darwin:boot:1789481317')).toBe('boot')
    expect(autoUpdateSessionScope('boot:5964937')).toBe('boot')
  })
})
