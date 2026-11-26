import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  backendTransitionFor,
  beginBackendTransition,
  resetBackendTransitionsForTests
} from '@/store/backend-transition'
import { registerGatewayReconnect } from '@/store/gateway-reconnect'
import { LAUNCHER_POLL_MS, stopLauncherWatchesForTests } from '@/store/launcher-transition'
import { $activeGatewayProfile, ensureGatewayProfile } from '@/store/profile'
import { $profileRemoteOverrides } from '@/store/profile-remote-override'
import { $gatewayState } from '@/store/session'

import { useLauncherTransitions } from './use-launcher-transitions'

// The hook is what ends a plugin-run profile's "Restarting sandbox…" when its connection is back.
// launcher-transition.test.ts covers the functions it calls; nothing covered that the hook calls them.

function Harness() {
  useLauncherTransitions()

  return null
}

beforeEach(() => {
  $gatewayState.set('idle')
  $activeGatewayProfile.set('work')
  $profileRemoteOverrides.set({
    work: { host: '127.0.0.1:47972', launcher: { plugin: 'acme' }, url: 'http://127.0.0.1:47972' }
  })
})

afterEach(() => {
  cleanup()
  stopLauncherWatchesForTests()
  resetBackendTransitionsForTests()
  vi.useRealTimers()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  $profileRemoteOverrides.set({})
  $activeGatewayProfile.set('default')
  $gatewayState.set('idle')
})

describe('useLauncherTransitions', () => {
  it("ends a plugin-run profile's restart state when the profile's socket opens", () => {
    render(<Harness />)
    beginBackendTransition('work', 'restarting', { source: 'launcher' })

    $gatewayState.set('open')

    expect(backendTransitionFor('work')).toBeNull()
  })

  it("leaves a switch's own state for the switch to end", () => {
    render(<Harness />)
    beginBackendTransition('work', 'starting', { source: 'switch' })

    $gatewayState.set('open')

    expect(backendTransitionFor('work')?.kind).toBe('starting')
  })
})

// When the plugin's launcher reports its backend running again, the window reconnects at once
// instead of waiting out its backoff. That reconnect closes and redials the socket the window
// shows at that moment, so it belongs to the profile that restarted and to nothing else.
describe('the reconnect when a restarted backend is ready', () => {
  // What the plugin's status route answers.
  let launcher = 'restarting'
  let reconnect = vi.fn(async () => undefined)
  let offReconnect = () => undefined as void

  beforeEach(() => {
    vi.useFakeTimers()
    launcher = 'restarting'
    reconnect = vi.fn(async () => undefined)
    offReconnect = registerGatewayReconnect(reconnect)
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api: vi.fn(async () => ({ state: launcher })) }
  })

  afterEach(() => offReconnect())

  // The socket of `work` drops and the launcher says the restart is planned.
  async function restartBegins() {
    render(<Harness />)
    $gatewayState.set('closed')
    await vi.advanceTimersByTimeAsync(0)
    expect(backendTransitionFor('work')?.kind).toBe('restarting')
  }

  async function launcherReportsRunning() {
    launcher = 'running'
    await vi.advanceTimersByTimeAsync(LAUNCHER_POLL_MS)
    await vi.advanceTimersByTimeAsync(0)
  }

  it('reconnects the window while it still shows the profile that restarted', async () => {
    await restartBegins()
    await launcherReportsRunning()

    expect(reconnect).toHaveBeenCalledTimes(1)
    expect(reconnect).toHaveBeenCalledWith({ source: 'restart-followthrough' })
  })

  it('leaves the window alone once it shows another profile: that chat is not cut', async () => {
    await restartBegins()
    // The person moves to another profile's chat while `work` restarts.
    $activeGatewayProfile.set('coder')
    await launcherReportsRunning()

    expect(reconnect).not.toHaveBeenCalled()
  })

  it('reconnects when a pick made meanwhile failed and the window still shows the profile', async () => {
    await restartBegins()
    // Nothing answers the route lookup here: the pick fails and the window stays on `work`.
    await ensureGatewayProfile('coder').catch(() => undefined)
    expect($activeGatewayProfile.get()).toBe('work')
    await launcherReportsRunning()

    expect(reconnect).toHaveBeenCalledTimes(1)
  })
})
