/**
 * backend-release-gate.test.ts
 *
 * The #74805 first-attempt race, pinned as a contract on the extracted gate:
 * the desktop must not hand off to the updater while PIDs it signalled are
 * still in the process table, even when the venv shim probe reads unlocked
 * (the backend `python.exe -m hermes_cli.main serve` need not hold the shim
 * at all). On merge-base main.ts the gate was shim-only and passed on its
 * first iteration with zero dwell — the sabotage A/B run proves these tests
 * bite on that behavior.
 */

import { describe, expect, it } from 'vitest'

import { RELEASE_GATE_POLL_MS, type ReleaseGateDeps, waitForBackendRelease } from './backend-release-gate'

/** A fake clock where sleep() advances time instantly. */
function fakeClock() {
  let t = 0

  return {
    now: () => t,
    sleep: async (ms: number) => {
      t += ms
    },
    advance: (ms: number) => {
      t += ms
    }
  }
}

function makeDeps(overrides: Partial<ReleaseGateDeps> = {}): ReleaseGateDeps & {
  logs: string[]
  kills: number[]
} {
  const clock = fakeClock()
  const logs: string[] = []
  const kills: number[] = []

  return {
    isShimLocked: () => false,
    isPidAlive: () => false,
    collectStragglerPids: () => [],
    killProcessTree: pid => kills.push(pid),
    sleep: clock.sleep,
    now: clock.now,
    log: line => logs.push(line),
    logs,
    kills,
    ...overrides
  }
}

describe('waitForBackendRelease (#74805 first-attempt race)', () => {
  it('does NOT pass while a signalled PID is still in the process table, even with the shim unlocked', async () => {
    // The exact #74805 shape: shim unlocked from tick 0 (serve backend never
    // held it), but the killed python is still tearing down for ~1.2s.
    let aliveUntil = 4 * RELEASE_GATE_POLL_MS
    const clock = fakeClock()

    const deps = makeDeps({
      now: clock.now,
      sleep: clock.sleep,
      isShimLocked: () => false,
      isPidAlive: () => clock.now() < aliveUntil
    })

    const result = await waitForBackendRelease([4021], deps, 'test')

    expect(result.unlocked).toBe(true)
    expect(result.lingeringPids).toEqual([])
    // The gate must have dwelled at least until the PID actually exited —
    // on merge-base (shim-only gate) it would have returned at t=0.
    expect(clock.now()).toBeGreaterThanOrEqual(aliveUntil)
  })

  it('passes immediately when the shim is unlocked and no signalled PID lingers', async () => {
    const deps = makeDeps()

    const result = await waitForBackendRelease([4021, 4022], deps, 'test')

    expect(result.unlocked).toBe(true)
    expect(deps.now()).toBe(0) // no dwell needed — everything already gone
  })

  it('keeps waiting while the shim is locked and fails closed at the deadline', async () => {
    const deps = makeDeps({ isShimLocked: () => true })

    const result = await waitForBackendRelease([], deps, 'test', 3 * RELEASE_GATE_POLL_MS)

    expect(result.unlocked).toBe(false)
  })

  it('proceeds at the deadline when the shim is unlocked but PIDs still linger (pre-#74805 escape hatch)', async () => {
    // Lingering PIDs past the deadline are the venv-blocker re-scan's job —
    // the gate must not invent a new failure mode for them.
    const deps = makeDeps({ isPidAlive: () => true })

    const result = await waitForBackendRelease([4021], deps, 'test', 3 * RELEASE_GATE_POLL_MS)

    expect(result.unlocked).toBe(true)
    expect(result.lingeringPids).toEqual([4021])
  })

  it('kills and then waits out stragglers that respawn mid-teardown', async () => {
    // A pool entry registered mid-teardown appears on pass 2; the gate must
    // signal it AND add it to the exit-wait set.
    const clock = fakeClock()
    let stragglerServed = false
    let stragglerKilledAt: number | null = null
    const kills: number[] = []

    const deps = makeDeps({
      now: clock.now,
      sleep: clock.sleep,
      collectStragglerPids: () => {
        if (!stragglerServed) {
          stragglerServed = true

          return [7777]
        }

        return []
      },
      killProcessTree: pid => {
        stragglerKilledAt = clock.now()
        kills.push(pid)
      },
      // Primary PID 4021 lingers for one poll (forcing a straggler-collect
      // pass); the straggler stays alive for two polls after being killed.
      isPidAlive: pid => {
        if (pid === 4021) {
          return clock.now() < RELEASE_GATE_POLL_MS
        }

        return pid === 7777 && stragglerKilledAt !== null && clock.now() < stragglerKilledAt + 2 * RELEASE_GATE_POLL_MS
      }
    })

    const result = await waitForBackendRelease([4021], deps, 'test')

    expect(kills).toContain(7777)
    expect(result.unlocked).toBe(true)
    expect(result.lingeringPids).toEqual([])
    // The gate must have dwelled until the straggler actually exited.
    expect(clock.now()).toBeGreaterThanOrEqual((stragglerKilledAt ?? 0) + 2 * RELEASE_GATE_POLL_MS)
  })

  it('ignores invalid PIDs in the seed and straggler sets', async () => {
    const deps = makeDeps({
      collectStragglerPids: () => [0, -4, NaN as unknown as number]
    })

    const result = await waitForBackendRelease(
      [0, -1, 2.5, NaN as unknown as number],
      deps,
      'test',
      2 * RELEASE_GATE_POLL_MS
    )

    expect(result.unlocked).toBe(true)
    expect(deps.kills).toEqual([])
  })
})
