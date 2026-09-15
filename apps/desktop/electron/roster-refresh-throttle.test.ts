import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it, vi } from 'vitest'

import { createRosterRefreshThrottle } from './roster-refresh-throttle'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

/** Deterministic clock: the throttle's cooldown/backoff windows are measured,
 *  never waited on, so no test sleeps. */
function fakeClock(start = 1_700_000_000_000) {
  let now = start

  return {
    now: () => now,
    advance: (ms: number) => {
      now += ms
    }
  }
}

describe('roster refresh throttle (#104227)', () => {
  it('coalesces concurrent refreshes onto ONE enumeration', async () => {
    const clock = fakeClock()
    const throttle = createRosterRefreshThrottle({ now: clock.now })
    let resolveEnumeration: ((value: string) => void) | undefined

    const enumerate = vi.fn(
      () =>
        new Promise<string>(resolve => {
          resolveEnumeration = resolve
        })
    )

    // Boot: the sidebar mount, the skills pane, and an SDK profile-routed
    // call can all ask for the roster before the first answer lands.
    const first = throttle.run(enumerate)
    const second = throttle.run(enumerate)
    const third = throttle.run(enumerate)

    expect(enumerate).toHaveBeenCalledTimes(1)
    expect(throttle.inFlight).toBe(true)

    resolveEnumeration?.('roster-1')

    expect(await Promise.all([first, second, third])).toEqual(['roster-1', 'roster-1', 'roster-1'])
    expect(enumerate).toHaveBeenCalledTimes(1)
    expect(throttle.inFlight).toBe(false)
  })

  it('serves the last snapshot inside the cooldown, then re-enumerates once it elapses', async () => {
    const clock = fakeClock()
    const throttle = createRosterRefreshThrottle({ now: clock.now, cooldownMs: 5_000 })
    let calls = 0
    const enumerate = async () => `roster-${++calls}`

    expect(await throttle.run(enumerate)).toBe('roster-1')

    // The sidebar's next mount/focus lands inside the retry window: no new
    // enumeration, the fresh snapshot is handed back instead.
    clock.advance(1_000)
    expect(await throttle.run(enumerate)).toBe('roster-1')
    expect(calls).toBe(1)

    clock.advance(5_000)
    expect(await throttle.run(enumerate)).toBe('roster-2')
    expect(calls).toBe(2)
  })

  it('a failed refresh backs off exponentially instead of re-hammering the pool', async () => {
    const clock = fakeClock()
    const throttle = createRosterRefreshThrottle({ now: clock.now, cooldownMs: 5_000, maxCooldownMs: 60_000 })
    let calls = 0

    const failing = () => {
      calls += 1

      return Promise.reject(new Error('pool cap: spawn refused'))
    }

    // Cold failure: nothing to serve, so the poll still retries at its own
    // cadence (fail open, never latched)...
    await expect(throttle.run(failing)).rejects.toThrow('pool cap: spawn refused')
    await expect(throttle.run(failing)).rejects.toThrow('pool cap: spawn refused')
    expect(calls).toBe(2)

    // ...but the cooldown doubles on every failure: 5s -> 10s -> 20s...
    expect(throttle.cooldownMs).toBe(20_000)

    // A warm snapshot is now held for the whole backoff window — the pool-cap
    // storm waits politely instead of respawning on every fetch.
    const healthy = async () => {
      calls += 1

      return 'recovered'
    }

    expect(await throttle.run(healthy)).toBe('recovered')
    expect(calls).toBe(3)
    expect(throttle.cooldownMs).toBe(5_000)
  })

  it('backoff window holds the warm snapshot for callers that arrive early', async () => {
    const clock = fakeClock()
    const throttle = createRosterRefreshThrottle({ now: clock.now, cooldownMs: 5_000, maxCooldownMs: 60_000 })
    let calls = 0
    let fail = false

    const enumerate = () => {
      calls += 1

      return fail ? Promise.reject(new Error('spawn refused')) : Promise.resolve(`roster-${calls}`)
    }

    expect(await throttle.run(enumerate)).toBe('roster-1')

    clock.advance(5_000)
    fail = true
    await expect(throttle.run(enumerate)).rejects.toThrow('spawn refused')
    expect(calls).toBe(2)
    expect(throttle.cooldownMs).toBe(10_000)

    // Inside the doubled window: the stale-but-good snapshot is served and the
    // backend that just refused a spawn is left alone.
    clock.advance(6_000)
    expect(await throttle.run(enumerate)).toBe('roster-1')
    expect(calls).toBe(2)

    clock.advance(4_500)
    fail = false
    expect(await throttle.run(enumerate)).toBe('roster-3')
    expect(calls).toBe(3)
  })

  it('a degraded (source-error) refresh backs off, and the next clean one resets the cooldown', async () => {
    const clock = fakeClock()
    const throttle = createRosterRefreshThrottle({ now: clock.now, cooldownMs: 5_000, maxCooldownMs: 60_000 })
    const degraded = (payload: { sources: { error?: string }[] }) => Boolean(payload?.sources?.some(source => source.error))
    let calls = 0

    const enumerate = async (): Promise<{ sources: Array<{ connectionId?: string; error?: string }> }> => {
      calls += 1

      return calls === 2
        ? { sources: [{ error: 'profile backend start timed out while waiting for a free slot.' }] }
        : { sources: [{ connectionId: 'local' }] }
    }

    expect(await throttle.run(enumerate, degraded)).toEqual({ sources: [{ connectionId: 'local' }] })
    expect(throttle.cooldownMs).toBe(5_000)

    // The enumeration RESOLVED, but a source came back refused: the pool cap
    // was hit, so the next refresh must wait longer, not immediately retry.
    clock.advance(5_000)
    await throttle.run(enumerate, degraded)
    expect(throttle.cooldownMs).toBe(10_000)
    expect(calls).toBe(2)

    clock.advance(5_000)
    await throttle.run(enumerate, degraded)
    expect(calls).toBe(2) // still inside the backed-off window: served stale
    expect(throttle.cooldownMs).toBe(10_000)

    clock.advance(5_000)
    await throttle.run(enumerate, degraded)
    expect(calls).toBe(3)
    expect(throttle.cooldownMs).toBe(5_000) // clean refresh resets the backoff
  })

  it('propagates a synchronous throw to the caller without wedging the seam', async () => {
    const clock = fakeClock()
    const throttle = createRosterRefreshThrottle({ now: clock.now })

    await expect(
      throttle.run(() => {
        throw new Error('registry unreadable')
      })
    ).rejects.toThrow('registry unreadable')

    expect(throttle.inFlight).toBe(false)
    expect(await throttle.run(async () => 'ok')).toBe('ok')
  })
})

describe('main.ts wiring for #104227', () => {
  it('routes the union roster IPC through the bounded refresh throttle', () => {
    const handlerStart = mainSource.indexOf("ipcMain.handle('hermes:agents:roster',")
    expect(handlerStart).toBeGreaterThan(-1)
    const body = mainSource.slice(handlerStart, handlerStart + 2_600)

    expect(body).toContain('rosterRefreshThrottle.run(')
    expect(body).toContain('enumerateRegistryAgentSources(')
    expect(body).toContain('connect-on-demand')
  })
})
