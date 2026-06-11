import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $desktopBoot } from '@/store/boot'
import { $desktopOnboarding } from '@/store/onboarding'
import { $gatewayState, setGatewayState } from '@/store/session'

import { BootFailureOverlay } from './boot-failure-overlay'
import { GatewayConnectingOverlay } from './gateway-connecting-overlay'

// Repro for the "remote gateway → stuck on CONNECTING, no way to settings"
// report. The connecting overlay (z-1200, full-screen, pointer-events on) is
// shown whenever `gatewayState !== 'open' && !boot.error`. The ONLY escape
// hatch — BootFailureOverlay, which has "Use local gateway" / "Sign in" /
// "Retry" — only renders when `boot.error` is set.
//
// useGatewayBoot only calls failDesktopBoot() (which sets boot.error) when the
// INITIAL boot() throws. After the first successful connect (bootCompleted),
// any later socket drop goes through scheduleReconnect(), which loops FOREVER
// against the dead remote and never sets boot.error. So gatewayState sits at
// 'closed'/'error' with boot.error null → CONNECTING forever, recovery overlay
// never appears, settings unreachable.

function resetStores() {
  setGatewayState('idle')
  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: 'ready',
    phase: 'renderer.ready',
    progress: 100,
    running: false,
    timestamp: Date.now(),
    visible: false
  })
  $desktopOnboarding.set({
    configured: true,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false
  })
}

beforeEach(resetStores)
afterEach(cleanup)

// The connecting overlay renders "CONN" + a scrambled tail inside one
// uppercase span; match that node specifically so the recovery overlay's
// "Lost connection…" copy doesn't read as a false positive.
const isConnectingShown = () =>
  screen.queryAllByText((_, el) => /^CONN[/\\|\-_=+<>~:*A-Z]*$/.test(el?.textContent?.trim() ?? '')).length > 0
const isRecoveryShown = () =>
  Boolean(screen.queryByText(/use local gateway/i) || screen.queryByText(/retry/i) || screen.queryByText(/sign in/i))

describe('connecting overlay vs recovery surface', () => {
  it('hard initial-boot failure surfaces the recovery overlay (the working path)', () => {
    // failDesktopBoot() ran: error set, gateway never opened.
    $desktopBoot.set({ ...$desktopBoot.get(), error: 'Hermes backend did not become ready', running: false, visible: true })
    setGatewayState('error')

    render(
      <>
        <GatewayConnectingOverlay />
        <BootFailureOverlay />
      </>
    )

    expect(isRecoveryShown()).toBe(true)
    // Connecting overlay bows out when boot.error is set.
    expect(isConnectingShown()).toBe(false)
  })

  it('REPRO: remote socket drops AFTER a successful boot → stuck on CONNECTING, no recovery, no settings', () => {
    // 1. Initial boot succeeded: gateway opened, boot completed (no error).
    setGatewayState('open')
    const { rerender } = render(
      <>
        <GatewayConnectingOverlay />
        <BootFailureOverlay />
      </>
    )
    expect(isConnectingShown()).toBe(false)

    // 2. The remote VPS socket drops (sleep/wake, remote restart, network).
    //    bootCompleted is true, so useGatewayBoot routes this through
    //    scheduleReconnect() — boot.error stays NULL.
    setGatewayState('closed')
    rerender(
      <>
        <GatewayConnectingOverlay />
        <BootFailureOverlay />
      </>
    )

    // The connecting overlay reappears and latches...
    expect(isConnectingShown()).toBe(true)
    // ...with NO recovery surface, because boot.error was never set.
    expect(isRecoveryShown()).toBe(false)

    // 3. Reconnect loops forever against the dead remote: gatewayState bounces
    //    closed → error → closed, boot.error never gets set. The user is
    //    pinned on CONNECTING with no path to Settings indefinitely.
    setGatewayState('error')
    rerender(
      <>
        <GatewayConnectingOverlay />
        <BootFailureOverlay />
      </>
    )
    expect($desktopBoot.get().error).toBeNull()
    expect(isConnectingShown()).toBe(true)
    expect(isRecoveryShown()).toBe(false)
  })

  it('FIX: once the prolonged reconnect raises a recoverable boot error, the recovery overlay takes over', () => {
    // Mirrors what useGatewayBoot.scheduleReconnect() now does after ~45s of
    // failed post-boot reconnects: it calls failDesktopBoot(), flipping the UI
    // from the dead-end CONNECTING overlay to the recovery surface.
    setGatewayState('error')
    $desktopBoot.set({
      ...$desktopBoot.get(),
      error: 'Lost connection to the Hermes gateway and could not reconnect.',
      running: false,
      visible: true
    })

    render(
      <>
        <GatewayConnectingOverlay />
        <BootFailureOverlay />
      </>
    )

    // Escape hatch is now reachable; the connecting overlay bows out.
    expect(isRecoveryShown()).toBe(true)
    expect(screen.getByText(/use local gateway/i)).toBeTruthy()
    expect(isConnectingShown()).toBe(false)
  })
})
