/**
 * #C3 (t_406dbcb3): the deferred post-connect refresh pair.
 *
 * The claim must be ONE per reconnect episode, fired only when the generation
 * is application-usable, and never awaited by the reconnect path. These are
 * the scheduler's own contracts; the wiring proof (the reconnect completes and
 * the lists are still in flight) lives in use-gateway-boot-deferred-refresh.
 */
import { describe, expect, it, vi } from 'vitest'

import { createPostConnectRefreshScheduler } from './use-gateway-boot-post-connect'

function harness(overrides: Partial<Parameters<typeof createPostConnectRefreshScheduler>[0]> = {}) {
  const refreshHermesConfig = vi.fn(async () => undefined)
  const refreshSessions = vi.fn(async () => undefined)
  let usableResolve: () => void = () => undefined
  const applicationUsable = vi.fn(
    () =>
      new Promise<void>(resolve => {
        usableResolve = resolve
      })
  )
  const episode = { id: 'ep-1' }
  let currentEpisode = vi.fn(() => episode as object | null)

  const scheduler = createPostConnectRefreshScheduler({
    refreshHermesConfig,
    refreshSessions,
    currentEpisode: () => currentEpisode(),
    applicationUsable,
    ...overrides
  })

  return { applicationUsable, currentEpisode, episode, refreshHermesConfig, refreshSessions, scheduler, usableResolve: () => usableResolve() }
}

const settled = () => new Promise(resolve => setTimeout(resolve, 0))

describe('createPostConnectRefreshScheduler', () => {
  it('does not fire the pair until the generation is application-usable', async () => {
    const test = harness()

    test.scheduler.schedule()
    await settled()

    expect(test.refreshHermesConfig).not.toHaveBeenCalled()

    test.usableResolve()
    await settled()

    expect(test.refreshHermesConfig).toHaveBeenCalledTimes(1)
  })

  it('runs config before sessions (the serial order the awaited path had)', async () => {
    const order: string[] = []
    let resolveConfig: () => void = () => undefined
    const configGate = new Promise<void>(resolve => {
      resolveConfig = resolve
    })

    const test = harness({
      refreshHermesConfig: async () => {
        order.push('config')
        await configGate
      },
      refreshSessions: async () => {
        order.push('sessions')
      }
    })

    test.scheduler.schedule()
    test.usableResolve()
    await settled()

    expect(order).toEqual(['config'])

    resolveConfig()
    await settled()

    expect(order).toEqual(['config', 'sessions'])
  })

  it('claims ONE pair per episode even when several finishes race', () => {
    const test = harness()

    test.scheduler.schedule()
    test.scheduler.schedule()
    test.scheduler.schedule()

    // One applicationUsable waiter registered, not one per finish.
    expect(test.applicationUsable).toHaveBeenCalledTimes(1)
  })

  it('drops a post-settle straggler finish (token null after a claim)', async () => {
    const test = harness()

    test.scheduler.schedule()
    test.usableResolve()
    await settled()
    expect(test.refreshHermesConfig).toHaveBeenCalledTimes(1)

    // Episode settled; the losing generation of the SAME burst finishes late.
    test.currentEpisode.mockReturnValue(null)
    test.scheduler.schedule()
    await settled()

    expect(test.applicationUsable).toHaveBeenCalledTimes(1)
    expect(test.refreshHermesConfig).toHaveBeenCalledTimes(1)
  })

  it('claims again for a NEW episode', async () => {
    const test = harness()

    test.scheduler.schedule()
    test.usableResolve()
    await settled()
    expect(test.refreshHermesConfig).toHaveBeenCalledTimes(1)

    test.currentEpisode.mockReturnValue({ id: 'ep-2' })
    test.scheduler.schedule()
    test.usableResolve()
    await settled()

    expect(test.refreshHermesConfig).toHaveBeenCalledTimes(2)
    expect(test.refreshSessions).toHaveBeenCalledTimes(2)
  })

  it('a replaced claim fires only once (the newest wins)', async () => {
    let resolveFirst: () => void = () => undefined
    let resolveSecond: () => void = () => undefined
    let calls = 0
    const refreshHermesConfig = vi.fn(async () => undefined)
    const episode = [{ id: 'ep-1' }, { id: 'ep-2' }][0]

    const scheduler = createPostConnectRefreshScheduler({
      refreshHermesConfig,
      refreshSessions: vi.fn(async () => undefined),
      currentEpisode: () => episode,
      applicationUsable: () => {
        calls += 1

        return new Promise<void>(resolve => {
          if (calls === 1) {
            resolveFirst = resolve
          } else {
            resolveSecond = resolve
          }
        })
      }
    })

    scheduler.schedule()
    // Same episode object: deduped — no second claim even if we schedule again.
    scheduler.schedule()
    expect(calls).toBe(1)
    resolveFirst()
    await settled()
    expect(refreshHermesConfig).toHaveBeenCalledTimes(1)
  })

  it('dispose before usable stops the pending claim from firing', async () => {
    const test = harness()

    test.scheduler.schedule()
    test.scheduler.dispose()
    test.usableResolve()
    await settled()

    expect(test.refreshHermesConfig).not.toHaveBeenCalled()
  })

  it('a rejected config refresh still runs the sessions refresh', async () => {
    const test = harness({
      refreshHermesConfig: vi.fn(async () => {
        throw new Error('config fetch failed')
      })
    })

    test.scheduler.schedule()
    test.usableResolve()
    await settled()

    expect(test.refreshSessions).toHaveBeenCalledTimes(1)
  })
})
