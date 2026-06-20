import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

// The REAL desktop app (DesktopController) — its AppShell already collapses to a
// drawer on narrow viewports, so mounting it on a phone gives the desktop chat
// experience, responsively. We only gate it behind connect/login.
import DesktopController from '@/app'

import type { ProbeResult } from '~bridge/auth'
import { $reauthNonce, loadTarget, setTarget, type GatewayTarget } from '~bridge/state'

import { ConnectScreen } from '~mobile/connect/ConnectScreen'
import { LoginScreen } from '~mobile/connect/LoginScreen'
import { MobileBehaviors } from '~mobile/mobile-behaviors'
import { initNativeChrome } from '~mobile/native-init'

type View = 'loading' | 'connect' | 'login' | 'connected'

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

  // A 401 demanding re-login bounces us back to the login form.
  useEffect(() => {
    if (reauthNonce > 0 && probe) setView('login')
  }, [reauthNonce, probe])

  async function onProbeResult(p: ProbeResult) {
    setProbe(p)
    if (p.authMode === 'token') {
      await commit({ baseUrl: p.baseUrl, authMode: 'token', provider: null })
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

  // Connected → hand off to the real desktop chat app, plus the mobile touch
  // adaptations (sidebar drawers, settings master-detail, Android back).
  return (
    <>
      <DesktopController />
      <MobileBehaviors />
    </>
  )
}
