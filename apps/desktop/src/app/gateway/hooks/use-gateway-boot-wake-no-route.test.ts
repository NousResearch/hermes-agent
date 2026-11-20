/**
 * The wake dial's last orphan path (t_e54b39e1, follow-up to t_de06ee1e R5).
 *
 * `dial()` opens its episode BEFORE it resolves the route, so every exit after
 * that line must settle it. R5 named two of the three (a successor that never
 * opens; a dial that threw). The third is the route that never resolved:
 * `resolveRoute()` returning null used to `return` with the episode still OPEN,
 * leaving the next unrelated `usable`/`beginReconnectEpisode` to date it on a
 * clock that was not its own — a bogus `detect_ms` and a `retrying` friction
 * record for a wait nobody had.
 *
 * The host is FAKED here (that seam exists for exactly this shape: the module's
 * route resolution is one of the four things only the boot hook can do). The
 * real episode bookkeeping, the real metrics store and the real log line are
 * what the assertions read.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import {
  beginReconnectEpisode,
  resetDesktopMetricsForTests,
  settleReconnectEpisode
} from '@/store/desktop-metrics'
import { unparkSecondaryReconnects } from '@/store/gateway-park'

import { parseReconnectEpisode } from '../../../test/gateway-boot-harness'

import type { WakeHost, WakeRoute } from './use-gateway-boot-wake'
import { createWakeRecovery } from './use-gateway-boot-wake'

type Episode = Record<string, string>

function episodesOf(lines: string[]): Episode[] {
  return lines.map(parseReconnectEpisode).filter((episode): episode is Episode => episode !== null)
}

/** A window holding a primary that reports OPEN, and a host a test can steer. */
function fakeHost() {
  const primary = { connectionState: 'open' as string }

  const host: WakeHost = {
    attach: vi.fn(() => () => undefined),
    finish: vi.fn(async () => undefined),
    noteReauth: vi.fn(),
    primary: () => primary as unknown as HermesGateway,
    promote: vi.fn(() => () => undefined),
    ready: () => true,
    redial: vi.fn(async () => undefined),
    resolveRoute: vi.fn(async (): Promise<null | WakeRoute> => null)
  }

  return { host, primary }
}

describe('useGatewayBoot wake dial with no route (t_e54b39e1)', () => {
  let lines: string[]

  beforeEach(() => {
    lines = []
    resetDesktopMetricsForTests()
    unparkSecondaryReconnects()
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      logLine: (line: string) => lines.push(line)
    }
  })

  afterEach(() => {
    resetDesktopMetricsForTests()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('settles the episode it opened as abandoned, and leaves nothing open', async () => {
    const { host } = fakeHost()
    const recovery = createWakeRecovery(host)

    await recovery.dial()

    // The live primary is still the one this wake meant to replace (nothing was
    // swapped), so the wake opened an episode and the null route abandons it.
    expect(host.redial).not.toHaveBeenCalled()

    const episodes = episodesOf(lines)

    expect(episodes).toHaveLength(1)
    expect(episodes[0].outcome).toBe('abandoned')
    expect(episodes[0].close_reason).toBe('wake')
    expect(episodes[0].parked_close).toBe('no')
    // The detection phase belongs to this dial: the line carries a real
    // detect_ms rather than the `n/a` of an episode nobody closed the detection
    // on.
    expect(Number(episodes[0].detect_ms)).toBeGreaterThanOrEqual(0)

    // NOTHING is left open. An unrelated episode that starts later reports
    // exactly itself — no `retrying` line settling the orphan on the wrong
    // clock, and no second `wake` line either.
    beginReconnectEpisode({ closeReason: 'network', visibilityAtClose: 'visible' })
    settleReconnectEpisode('usable')

    expect(lines.filter(line => line.includes('outcome=retrying'))).toEqual([])

    const after = episodesOf(lines)

    expect(after.map(episode => episode.close_reason)).toEqual(['wake', 'network'])
    expect(after.map(episode => episode.outcome)).toEqual(['abandoned', 'usable'])
  })

  it('a second wake dial on the same window is not swallowed by an orphan', async () => {
    const { host } = fakeHost()
    const recovery = createWakeRecovery(host)

    await recovery.dial()
    await recovery.dial()

    // Two dials, two settled episodes: a leftover open episode would have made
    // the second `ensureReconnectEpisode` a no-op and reported nothing at all.
    expect(host.resolveRoute).toHaveBeenCalledTimes(2)
    expect(episodesOf(lines).map(episode => episode.outcome)).toEqual(['abandoned', 'abandoned'])
  })

  it('a primary that is not open still takes the redial branch, with no episode of its own', async () => {
    const { host, primary } = fakeHost()

    primary.connectionState = 'closed'

    const recovery = createWakeRecovery(host)

    await recovery.dial()

    // The redial branch owns its own episode (the retry ladder's, which must
    // stay open across rungs): the wake module never opens or settles one here.
    expect(host.redial).toHaveBeenCalledTimes(1)
    expect(host.resolveRoute).not.toHaveBeenCalled()
    expect(episodesOf(lines)).toEqual([])
  })
})
