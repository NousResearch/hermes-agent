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

  it('trusts the backend "no platforms configured" verdict over a retained stopped state', () => {
    // A Desktop install that never set up messaging: the gateway is stopped (or was stopped
    // by the last app quit) and projects an empty platform map. There is no bot to be
    // missing, so the pill must not alarm.
    const neverConfigured = statusBarGatewayHealth({
      ...openReady,
      messagingConfigured: false,
      messagingRunning: false,
      messagingState: 'stopped',
      platforms: {}
    })

    expect(`${neverConfigured.label} ${neverConfigured.detail}`).toBe(`${copy.backend} ${copy.ready}`)
    expect(neverConfigured.degraded).toBe(false)

    // With platforms configured, the same stopped state is a real outage and still alarms.
    const configuredDown = statusBarGatewayHealth({
      ...openReady,
      messagingConfigured: true,
      messagingRunning: false,
      messagingState: 'stopped',
      platforms: {}
    })

    expect(configuredDown.detail).toBe(copy.messagingStopped)
    expect(configuredDown.degraded).toBe(true)

    // An unknown verdict (older backend, unreadable config) keeps the pre-verdict behavior.
    const unknown = statusBarGatewayHealth({
      ...openReady,
      messagingConfigured: null,
      messagingRunning: false,
      messagingState: 'stopped',
      platforms: {}
    })

    expect(unknown.detail).toBe(copy.messagingStopped)
    expect(unknown.degraded).toBe(true)

    // A configured=false verdict never outranks a non-empty platform map: merged per-profile
    // entries carry real failures the local config set cannot see.
    const mergedFailure = statusBarGatewayHealth({
      ...openReady,
      messagingConfigured: false,
      messagingRunning: true,
      messagingState: 'running',
      platforms: { 'work:telegram': { state: 'fatal' } }
    })

    expect(mergedFailure.detail).toBe(copy.messagingDegraded('telegram'))
    expect(mergedFailure.degraded).toBe(true)

    // startup_failed is an actionable failure, not a plain stop: it alarms even when the
    // config verdict says nothing is configured and the map is empty (fatal entries are
    // only projected for startup_failed, and the verdict must not swallow them).
    const startupFailed = statusBarGatewayHealth({
      ...openReady,
      messagingConfigured: false,
      messagingRunning: false,
      messagingState: 'startup_failed',
      platforms: {}
    })

    expect(startupFailed.detail).toBe(copy.messagingStopped)
    expect(startupFailed.degraded).toBe(true)

    // Case/whitespace variants normalize before the verdict check.
    const messy = statusBarGatewayHealth({
      ...openReady,
      messagingConfigured: false,
      messagingRunning: false,
      messagingState: ' Stopped ',
      platforms: {}
    })

    expect(messy.detail).toBe(copy.ready)
    expect(messy.degraded).toBe(false)

    // A watchdog "degraded" record with nothing configured does not alarm the messaging pill
    // (unchanged from pre-verdict behavior): components.gateway.status still reports the
    // degraded verdict; the messaging pill only speaks about platforms.
    const degradedQuiet = statusBarGatewayHealth({
      ...openReady,
      messagingConfigured: false,
      messagingRunning: false,
      messagingState: 'degraded',
      platforms: {}
    })

    expect(degradedQuiet.detail).toBe(copy.ready)
    expect(degradedQuiet.degraded).toBe(false)
  })
})
