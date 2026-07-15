import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

// The REAL desktop app (DesktopController) — its AppShell already collapses to a
// drawer on narrow viewports, so mounting it on a phone gives the desktop chat
// experience, responsively. We only gate it behind connect/login.
import DesktopController from '@/app'
import { $desktopBoot } from '@/store/boot'

import type { ProbeResult } from '~bridge/auth'
import { $reauthNonce, loadTarget, setTarget, type GatewayTarget } from '~bridge/state'

import { ConnectScreen } from '~mobile/connect/ConnectScreen'
import { LoginScreen } from '~mobile/connect/LoginScreen'
import { TokenScreen } from '~mobile/connect/TokenScreen'
import { MobileBehaviors } from '~mobile/mobile-behaviors'
import { initNativeChrome } from '~mobile/native-init'

type View = 'loading' | 'connect' | 'login' | 'token' | 'connected'

export function MobileRoot() {
  const [view, setView] = useState<View>('loading')
  const [probe, setProbe] = useState<ProbeResult | null>(null)
  const reauthNonce = useStore($reauthNonce)

  // Boot: configure native chrome (status bar), then restore a saved gateway
  // target, else show the connect screen.
  useEffect(() => {
    void initNativeChrome()
    void (async () => {
      const t = await loadTarget()
      setView(t ? 'connected' : 'connect')
    })()
  }, [])

  // A 401 demanding re-login bounces us back to the right entry form for the
  // gateway's auth mode (token gateways re-collect the token, not a password).
  useEffect(() => {
    if (reauthNonce > 0 && probe) setView(probe.authMode === 'token' ? 'token' : 'login')
  }, [reauthNonce, probe])

  async function onProbeResult(p: ProbeResult) {
    setProbe(p)
    if (p.authMode === 'token') {
      // Token gateways need a static session token; collect it before committing
      // (an empty token is rejected by both the gateway and desktop parity).
      setView('token')
      return
    }
    if (!p.needsLogin) {
      const provider = p.providers.find((x) => x.supportsPassword)?.name ?? 'basic'
      await commit({ baseUrl: p.baseUrl, authMode: 'oauth', provider })
      return
    }
    setView('login')
  }

  async function commit(t: GatewayTarget) {
    await setTarget(t)
    setView('connected')
  }

  if (view === 'loading') {
    return (
      <div className="flex min-h-full items-center justify-center text-sm text-muted-foreground">
        Loading…
      </div>
    )
  }

  if (view === 'connect') {
    return <ConnectScreen initialUrl={probe?.baseUrl ?? ''} onResult={onProbeResult} />
  }

  if (view === 'login' && probe) {
    return (
      <LoginScreen
        probe={probe}
        onBack={() => setView('connect')}
        onLoggedIn={(provider) => commit({ baseUrl: probe.baseUrl, authMode: 'oauth', provider })}
      />
    )
  }

  if (view === 'token' && probe) {
    return (
      <TokenScreen
        probe={probe}
        onBack={() => setView('connect')}
        onToken={(token) =>
          commit({ baseUrl: probe.baseUrl, authMode: 'token', provider: null, token })
        }
      />
    )
  }

  // Connected → hand off to the real desktop chat app, plus the mobile touch
  // adaptations (sidebar drawers, settings master-detail, Android back).
  return (
    <>
      <DesktopController />
      <MobileBehaviors />
      <BootAutoRetry />
    </>
  )
}

/**
 * Cold-launch connections sometimes time out because the Wi-Fi radio is still
 * waking from power-save (the foreground lock hasn't ramped it yet) — a reload a
 * second later, with the radio warm, connects fine. Rather than greet the user
 * with the failure screen, quietly reload-retry up to twice behind a
 * "Reconnecting…" cover; only after that does the manual BootFailureOverlay take
 * over. The counter lives in sessionStorage so it survives the reload, and is
 * cleared once the gateway connects.
 */
function BootAutoRetry() {
  const boot = useStore($desktopBoot)
  const [retrying, setRetrying] = useState(false)
  const scheduled = useRef(false)

  useEffect(() => {
    const KEY = 'hermes-boot-retries'
    const MAX = 2

    if (boot.running && !boot.error) {
      window.sessionStorage.removeItem(KEY)
      return
    }

    if (boot.error && !boot.running && !scheduled.current) {
      const tries = Number(window.sessionStorage.getItem(KEY) || '0')
      if (tries >= MAX) return // give up — let BootFailureOverlay handle it
      scheduled.current = true
      window.sessionStorage.setItem(KEY, String(tries + 1))
      setRetrying(true)
      const timer = window.setTimeout(() => {
        void window.hermesDesktop?.resetBootstrap?.()
        window.location.reload()
      }, 1100)
      return () => window.clearTimeout(timer)
    }
  }, [boot.running, boot.error])

  if (!retrying) return null
  return (
    <div className="fixed inset-0 z-[1500] grid place-items-center bg-(--ui-chat-surface-background) text-sm text-muted-foreground">
      Reconnecting…
    </div>
  )
}
