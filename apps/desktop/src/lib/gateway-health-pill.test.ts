import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { statusBarGatewayHealth } from './gateway-health-pill'

const copy = {
  backend: en.shell.statusbar.backend,
  checking: en.shell.statusbar.gatewayChecking,
  connecting: en.shell.statusbar.gatewayConnecting,
  messagingDegraded: en.shell.statusbar.messagingDegraded,
  messagingStopped: en.shell.statusbar.messagingStopped,
  needsSetup: en.shell.statusbar.gatewayNeedsSetup,
  offline: en.shell.statusbar.gatewayOffline,
  ready: en.shell.statusbar.gatewayReady,
  restarting: en.shell.statusbar.gatewayRestarting,
  unavailable: en.shell.statusbar.gatewayUnavailable
}

const inferenceReady = {
  checksDisagree: false,
  ready: true,
  reason: null,
  source: 'runtime_check' as const
}

const openReady = {
  connectionState: 'open',
  copy,
  inferenceStatus: inferenceReady
}

describe('statusBarGatewayHealth', () => {
  it('does not paint Gateway ready when the serve socket is up and messaging is down', () => {
    const pill = statusBarGatewayHealth({
      ...openReady,
      messagingRunning: false,
      messagingState: 'stopped',
      platforms: {}
    })

    expect(`${pill.label} ${pill.detail}`).not.toBe('Gateway ready')
    expect(pill.label).toBe(copy.backend)
    expect(pill.detail).toBe(copy.messagingStopped)
    expect(pill.degraded).toBe(true)
  })

  it('stays backend-ready when messaging was never configured, and names a down platform while the process is up', () => {
    const quiet = statusBarGatewayHealth({
      ...openReady,
      messagingRunning: false,
      messagingState: null,
      platforms: {}
    })

    expect(`${quiet.label} ${quiet.detail}`).toBe(`${copy.backend} ${copy.ready}`)
    expect(quiet.degraded).toBe(false)

    const discordDown = statusBarGatewayHealth({
      ...openReady,
      messagingRunning: true,
      messagingState: 'running',
      platforms: { discord: { state: 'fatal' } }
    })

    expect(discordDown.detail).toBe(copy.messagingDegraded('discord'))
    expect(discordDown.degraded).toBe(true)
  })
})
