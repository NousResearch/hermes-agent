import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider, TRANSLATIONS } from '@/i18n'
import { $desktopBoot } from '@/store/boot'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { $desktopOnboarding } from '@/store/onboarding'
import { $introView } from '@/store/onboarding-intro'
import { markOnboardingDecided } from '@/store/onboarding-presence'
import { setGatewayState } from '@/store/session'

import { BootFailureOverlay } from './boot-failure-overlay'
import { GatewayConnectingOverlay } from './gateway-connecting-overlay'

// Repro for the "remote gateway → stuck on CONNECTING, no way to settings"
// report. The connecting overlay (full-screen, pointer-events on) used
// to be shown whenever `gatewayState !== 'open' && !boot.error`. The ONLY escape
// hatch — BootFailureOverlay, which has "Use local gateway" / "Sign in" /
// "Retry" — only renders when `boot.error` is set.
//
// useGatewayBoot only calls failDesktopBoot() (which sets boot.error) when the
// INITIAL boot() throws. After the first successful connect (bootCompleted),
// any later socket drop goes through scheduleReconnect(), which loops FOREVER
// against the dead remote. So gatewayState sits at 'closed'/'error' with
// boot.error null. The fix keeps the initial-boot overlay out of post-boot
// reconnects, leaving chat/settings usable while the reconnect loop runs.

function resetStores() {
  setGatewayState('idle')
  $gatewaySwitching.set(false)
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
    manual: false,
    localEndpoint: false,
    freeTierReady: false
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
  it('post-boot socket drops do not re-cover the app with the initial CONNECTING overlay', async () => {
    // 1. Initial boot succeeded: gateway opened, boot completed (no error).
    setGatewayState('open')

    let rerender!: (ui: React.ReactElement) => void
    await act(async () => {
      const result = render(
        <>
          <GatewayConnectingOverlay />
          <BootFailureOverlay />
        </>
      )

      rerender = result.rerender
    })

    expect(isConnectingShown()).toBe(false)

    // 2. The remote VPS socket drops (sleep/wake, remote restart, network).
    //    bootCompleted is true, so useGatewayBoot routes this through
    //    scheduleReconnect() — boot.error stays NULL.
    await act(async () => {
      setGatewayState('closed')
      rerender!(
        <>
          <GatewayConnectingOverlay />
          <BootFailureOverlay />
        </>
      )
    })

    // The initial-boot connecting overlay stays out of the way, so settings and
    // the composer remain reachable during the reconnect loop.
    expect(isConnectingShown()).toBe(false)
    expect(isRecoveryShown()).toBe(false)

    // 3. Reconnect loops against the dead remote: gatewayState bounces closed
    //    → error → closed. Until the escalation path sets boot.error, the app
    //    remains usable instead of modal-blocked.
    await act(async () => {
      setGatewayState('error')
      rerender!(
        <>
          <GatewayConnectingOverlay />
          <BootFailureOverlay />
        </>
      )
    })
    expect($desktopBoot.get().error).toBeNull()
    expect(isConnectingShown()).toBe(false)
    expect(isRecoveryShown()).toBe(false)
  })

  it('soft gateway switch keeps the shell — no fullscreen CONNECTING', async () => {
    setGatewayState('open')

    const { rerender } = render(
      <>
        <GatewayConnectingOverlay />
        <BootFailureOverlay />
      </>
    )

    await act(async () => {
      $gatewaySwitching.set(true)
      $desktopBoot.set({
        ...$desktopBoot.get(),
        running: true,
        visible: true,
        progress: 4,
        error: null
      })
      setGatewayState('closed')
      rerender(
        <>
          <GatewayConnectingOverlay />
          <BootFailureOverlay />
        </>
      )
    })

    expect(isConnectingShown()).toBe(false)
    expect(isRecoveryShown()).toBe(false)
  })

  it('FIX: confirmed reauth boot.error still surfaces the recovery overlay (not CONNECTING)', async () => {
    // Transport blips no longer set boot.error (toast + background retry).
    // Confirmed OAuth reauth still does — and that recovery surface must win
    // over any residual connecting state so Sign-in / Gateway settings stay reachable.
    setGatewayState('error')
    $desktopBoot.set({
      ...$desktopBoot.get(),
      error: 'Your remote gateway session has expired. Sign in again.',
      running: false,
      visible: true
    })

    await act(async () => {
      render(
        <>
          <GatewayConnectingOverlay />
          <BootFailureOverlay />
        </>
      )
    })

    // Escape hatch is reachable; the connecting overlay bows out.
    expect(isRecoveryShown()).toBe(true)
    expect(screen.getByRole('button', { name: /gateway settings/i })).toBeTruthy()
    expect(isConnectingShown()).toBe(false)
  })
})

describe('first-run starting screen', () => {
  afterEach(() => {
    $introView.set('off')
    vi.useRealTimers()
  })

  it('comes back over a booted app while Run onboarding again opens its chat, then leaves', async () => {
    vi.useFakeTimers()
    setGatewayState('open')
    markOnboardingDecided()

    await act(async () => {
      render(<GatewayConnectingOverlay />)
    })
    expect(isConnectingShown()).toBe(false)

    await act(async () => $introView.set('starting'))
    expect(isConnectingShown()).toBe(true)

    await act(async () => $introView.set('intro'))

    // The exit choreography: text out, then the overlay fades.
    for (let step = 0; step < 3; step += 1) {
      await act(async () => vi.advanceTimersByTime(1_000))
    }

    expect(isConnectingShown()).toBe(false)
  })
})

describe('locale', () => {
  it('reads the boot label from the catalogue instead of a hardcoded English string', async () => {
    setGatewayState('idle')
    $desktopBoot.set({
      ...$desktopBoot.get(),
      error: null,
      phase: 'backend.start',
      progress: 12,
      running: true,
      visible: true
    })

    // Only the first four characters are readable while the tail decodes, and
    // they already tell the catalogues apart: the Spanish "CONECTANDO" keeps
    // CONE where the hardcoded English "CONNECTING" kept CONN.
    const spanishLabel = () =>
      screen.queryAllByText((_, el) => /^CONE[/\\|\-_=+<>~:*A-Z]*$/.test(el?.textContent?.trim() ?? '')).length > 0

    await act(async () => {
      render(
        <I18nProvider configClient={null} initialLocale="es">
          <GatewayConnectingOverlay />
        </I18nProvider>
      )
    })

    expect(spanishLabel()).toBe(true)
    expect(TRANSLATIONS.es.common.connecting).toBe('Conectando')
  })
})
