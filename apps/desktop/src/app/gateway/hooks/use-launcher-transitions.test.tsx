import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { backendTransitionFor, beginBackendTransition, resetBackendTransitionsForTests } from '@/store/backend-transition'
import { $activeGatewayProfile } from '@/store/profile'
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
  resetBackendTransitionsForTests()
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
