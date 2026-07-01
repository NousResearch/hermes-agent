import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { DesktopLoginScreen } from '@/components/desktop-login-screen'
import { useI18n } from '@/i18n'
import { $authState, handleAuthGate, refreshAuthStatus } from '@/store/auth'
import type { OnboardingContext } from '@/store/onboarding'

interface DesktopAuthGateProps {
  /** True once the local backend/env is ready (gatewayState === 'open'), mirroring
   *  the onboarding overlay's gate. The login-state check only runs after env is
   *  ready, so on a first launch the env/install overlay shows first, THEN the
   *  login screen — never both at once, never the login screen before env is up. */
  enabled: boolean
  onSignedIn?: () => void
  requestGateway: OnboardingContext['requestGateway']
}

// The hard auth gate. Boot order: (a) env-ready gate = the existing bootstrap /
// gateway overlays (unchanged), then (b) THIS login-state check. When the user
// isn't signed in (or the account is abnormal), it renders a full-window login
// screen that BLOCKS the chat UI; only a successful sign-in lets the app through.
// It also wires the continuous auth gate: a 401 / 403 account_disabled from any
// backend call flips the state back to signed-out and the login screen retakes
// the window.
export function DesktopAuthGate({ enabled, onSignedIn, requestGateway }: DesktopAuthGateProps) {
  const { t } = useI18n()
  const { gateReason, status } = useStore($authState)

  // Continuous auth gate: subscribe once to the main-process broadcast so a lost
  // login / disabled account anywhere in the app returns the user here. Mounted
  // unconditionally (independent of `enabled`) so a mid-session 401 is caught
  // even while an overlay is up.
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onAuthGate?.(handleAuthGate)

    return () => unsubscribe?.()
  }, [])

  // Run the login-state check once env is ready. Re-run when env flips to ready
  // so a fresh install lands on the login screen right after bootstrap finishes.
  useEffect(() => {
    if (enabled) {
      void refreshAuthStatus()
    }
  }, [enabled])

  // Env not ready yet, or already signed in (real or managed-disabled build) →
  // don't gate: the boot/env overlays cover the pre-ready phase, and a signed-in
  // user goes straight to chat.
  if (!enabled || status === 'signed-in') {
    return null
  }

  // Env is ready but the first login-state check hasn't resolved yet. Cover the
  // window with the login surface (no buttons) so the chat never flashes before
  // we know whether to gate — resolves within one status() round-trip, then this
  // swaps to the real login screen or unmounts. A returning user is seeded
  // 'signed-in' from cache and never reaches here.
  if (status === 'checking') {
    return <div className="fixed inset-0 z-1400 bg-(--ui-chat-surface-background) [-webkit-app-region:drag]" />
  }

  // 'disabled' → account abnormal; a 'signed-out' that followed a 401 gate →
  // session-expired copy; a clean first-run 'signed-out' shows just the buttons.
  const notice =
    status === 'disabled'
      ? t.auth.login.accountDisabled
      : gateReason === 'unauthorized'
        ? t.auth.login.sessionExpired
        : null

  return <DesktopLoginScreen gateNotice={notice} onSignedIn={onSignedIn} requestGateway={requestGateway} />
}
