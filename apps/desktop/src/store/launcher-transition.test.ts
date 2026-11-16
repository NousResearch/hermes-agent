import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ hermesApi: vi.fn() }))

vi.mock('@/api/client', () => ({ hermesApi: api.hermesApi }))

import { backendTransitionFor, beginBackendTransition, resetBackendTransitionsForTests } from './backend-transition'
import {
  endLauncherTransitionOnOpen,
  LAUNCHER_POLL_MS,
  LAUNCHER_REOPEN_GRACE_MS,
  launcherReading,
  probeLauncherTransition,
  stopLauncherWatchesForTests
} from './launcher-transition'
import { $profileRemoteOverrides } from './profile-remote-override'

const launched = { host: '127.0.0.1:47972', launcher: { plugin: 'openshell' }, url: 'http://127.0.0.1:47972' }

beforeEach(() => {
  vi.useFakeTimers()
  $profileRemoteOverrides.set({ work: launched })
  api.hermesApi.mockReset()
})

afterEach(() => {
  stopLauncherWatchesForTests()
  resetBackendTransitionsForTests()
  $profileRemoteOverrides.set({})
  vi.useRealTimers()
})

describe("the launcher's status decides whether a drop is planned", () => {
  it('reads restarting/starting/stopping as planned, running as ready, anything else as not planned', () => {
    for (const state of ['restarting', 'starting', 'stopping'] as const) {
      expect(launcherReading({ state })).toEqual({ kind: state, planned: true })
    }

    expect(launcherReading({ state: 'running' })).toEqual({ planned: false, ready: true })
    expect(launcherReading({ state: 'stopped' })).toEqual({ planned: false, ready: false })
    expect(launcherReading({})).toEqual({ planned: false, ready: false })
    expect(launcherReading(null)).toEqual({ planned: false, ready: false })
  })

  it('a restarting launcher publishes the transition, and its return ends it and reconnects', async () => {
    api.hermesApi.mockResolvedValueOnce({ state: 'restarting' })
    const onReady = vi.fn()

    await expect(probeLauncherTransition('work', onReady)).resolves.toBe(true)
    expect(api.hermesApi).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/plugins/openshell/status', profile: 'work' })
    )
    expect(backendTransitionFor('work')).toMatchObject({ kind: 'restarting', source: 'launcher' })

    api.hermesApi.mockResolvedValueOnce({ state: 'restarting' })
    await vi.advanceTimersByTimeAsync(LAUNCHER_POLL_MS)
    expect(backendTransitionFor('work')?.kind).toBe('restarting')
    expect(onReady).not.toHaveBeenCalled()

    api.hermesApi.mockResolvedValueOnce({ state: 'running' })
    await vi.advanceTimersByTimeAsync(LAUNCHER_POLL_MS)
    // Back: reconnect now, and the planned state holds until the socket opens
    // (no offline/connecting flash between "running" and the open socket).
    expect(onReady).toHaveBeenCalledTimes(1)
    expect(backendTransitionFor('work')?.kind).toBe('restarting')
    // A drop seen meanwhile starts no second poll.
    await expect(probeLauncherTransition('work', onReady)).resolves.toBe(true)
    await vi.advanceTimersByTimeAsync(LAUNCHER_POLL_MS * 2)
    expect(api.hermesApi).toHaveBeenCalledTimes(3)

    endLauncherTransitionOnOpen('work')
    expect(backendTransitionFor('work')).toBeNull()
  })

  it('a socket that never re-opens after the backend is back ends the planned state after a grace', async () => {
    api.hermesApi.mockResolvedValueOnce({ state: 'restarting' })
    await probeLauncherTransition('work')
    api.hermesApi.mockResolvedValueOnce({ state: 'running' })
    await vi.advanceTimersByTimeAsync(LAUNCHER_POLL_MS)
    expect(backendTransitionFor('work')).not.toBeNull()

    await vi.advanceTimersByTimeAsync(LAUNCHER_REOPEN_GRACE_MS)
    expect(backendTransitionFor('work')).toBeNull()
  })

  it('a launcher that reports a failure, or stops answering, ends the transition so the error shows', async () => {
    api.hermesApi.mockResolvedValueOnce({ state: 'restarting' })
    await probeLauncherTransition('work')
    api.hermesApi.mockResolvedValueOnce({ state: 'stopped' })
    await vi.advanceTimersByTimeAsync(LAUNCHER_POLL_MS)
    expect(backendTransitionFor('work')).toBeNull()

    api.hermesApi.mockResolvedValueOnce({ state: 'restarting' })
    await probeLauncherTransition('work')
    api.hermesApi.mockRejectedValue(new Error('ECONNREFUSED'))
    await vi.advanceTimersByTimeAsync(LAUNCHER_POLL_MS * 4)
    expect(backendTransitionFor('work')).not.toBeNull()
    await vi.advanceTimersByTimeAsync(LAUNCHER_POLL_MS)
    expect(backendTransitionFor('work')).toBeNull()
  })

  it('never asks for a profile no plugin runs, and an unplanned answer starts nothing', async () => {
    await expect(probeLauncherTransition('default')).resolves.toBe(false)
    expect(api.hermesApi).not.toHaveBeenCalled()

    api.hermesApi.mockResolvedValueOnce({ state: 'running' })
    await expect(probeLauncherTransition('work')).resolves.toBe(false)
    expect(backendTransitionFor('work')).toBeNull()
  })

  it("leaves a switch's own state alone: no launcher poll takes it over", async () => {
    beginBackendTransition('work', 'stopping', { source: 'switch' })

    await expect(probeLauncherTransition('work')).resolves.toBe(true)
    await vi.advanceTimersByTimeAsync(LAUNCHER_POLL_MS * 3)
    expect(api.hermesApi).not.toHaveBeenCalled()
    expect(backendTransitionFor('work')).toMatchObject({ kind: 'stopping', source: 'switch' })
  })
})
