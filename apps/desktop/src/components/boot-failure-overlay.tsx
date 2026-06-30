import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { DesktopConnectionConfig } from '@/global'
import { useI18n } from '@/i18n'
import { $desktopBoot, completeDesktopBoot } from '@/store/boot'
import { notify, notifyError } from '@/store/notifications'
import { $desktopOnboarding } from '@/store/onboarding'

import type { RemoteReauth } from './boot-failure-reauth'
import { deriveProviderShape, isRemoteReauthFailure, signInLabel } from './boot-failure-reauth'
import { GatewayRecoveryPanel, GatewaySetupPanel } from './gateway-launcher-overlay'
import { classifyGatewayStartup, type GatewayStartupState } from './gateway-launcher-state'

type BusyAction = 'retry' | 'signin' | null

// A remote gateway whose access cookie has lapsed (e.g. the dashboard
// restarted on the remote box) boots into this overlay with a reauth-shaped
// error. The only fixes are to re-establish the remote session or update the
// configured gateway URL/token. The detection + copy helpers live in
// ./boot-failure-reauth so they're unit-testable without a React render.

// Recovery surface for a hard boot failure. Without this the app shell renders
// dead — "gateway offline", no composer, only a toast — with no way to retry,
// configure the gateway, or find the logs.
export function BootFailureOverlay() {
  const boot = useStore($desktopBoot)
  const onboarding = useStore($desktopOnboarding)
  const { t } = useI18n()
  const [busy, setBusy] = useState<BusyAction>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [showLogs, setShowLogs] = useState(false)
  const [remoteReauth, setRemoteReauth] = useState<RemoteReauth | null>(null)
  const [startup, setStartup] = useState<GatewayStartupState | null>(null)

  const visible = Boolean(boot.error) && !boot.running
  // While first-run onboarding owns the picker/flow we let it surface its own
  // progress; the recovery overlay is for hard failures, which it covers via a
  // higher z-index regardless of onboarding state.
  const suppressed = onboarding.flow.status !== 'idle' && onboarding.flow.status !== 'error'

  useEffect(() => {
    if (!visible) {
      return
    }

    void window.hermesDesktop
      ?.getRecentLogs()
      .then(res => setLogs(res.lines ?? []))
      .catch(() => undefined)
  }, [boot.error, visible])

  // Classify startup through the remote-client lens:
  // missing saved URL -> first-run setup; saved URL + auth failure -> sign-in;
  // saved URL + network/unexpected failure -> configured gateway recovery.
  useEffect(() => {
    if (!visible) {
      setRemoteReauth(null)
      setStartup(null)

      return
    }

    let cancelled = false

    void (async () => {
      const desktop = window.hermesDesktop

      let config: DesktopConnectionConfig | null = null

      try {
        config = desktop?.getConnectionConfig ? await desktop.getConnectionConfig() : null
      } catch {
        config = null
      }

      const next = classifyGatewayStartup(config, boot.error)

      if (cancelled) {
        return
      }

      setStartup(next)
      setRemoteReauth(null)

      if (!config || !isRemoteReauthFailure(config, boot.error)) {
        return
      }

      // Best-effort probe for the provider shape so the button copy matches
      // what the user will see in the login window (password form vs OAuth
      // redirect). Probe failure just keeps the generic copy.
      let shape = deriveProviderShape(null)

      try {
        const probe = await desktop?.probeConnectionConfig?.(config.remoteUrl)
        shape = deriveProviderShape(probe?.providers)
      } catch {
        // Generic copy is fine.
      }

      if (!cancelled) {
        setRemoteReauth({ url: config.remoteUrl, ...shape })
      }
    })()

    return () => {
      cancelled = true
    }
  }, [boot.error, visible])

  if (!visible || suppressed) {
    return null
  }

  const retry = async () => {
    setBusy('retry')
    window.location.reload()
  }

  const openGatewaySettings = () => {
    completeDesktopBoot()
    window.location.hash = '/settings?tab=gateway'
  }

  // Open the gateway's login window (renders the username/password form for a
  // basic gateway, or the OAuth redirect otherwise — the desktop drives both
  // through the same window). On a successful sign-in the session cookie is
  // re-established in the persistent partition; reload so boot re-runs and the
  // reconnect now mints a ticket against a live session.
  const signInRemote = async () => {
    const url = remoteReauth?.url || startup?.remoteUrl

    if (!url) {
      return
    }

    setBusy('signin')

    try {
      const result = await window.hermesDesktop?.oauthLoginConnectionConfig(url)

      if (result?.connected) {
        notify({ kind: 'success', title: t.boot.failure.signedInTitle, message: t.boot.failure.signedInMessage })
        window.location.reload()

        return
      }

      notify({
        kind: 'warning',
        title: t.boot.failure.signInIncompleteTitle,
        message: t.boot.failure.signInIncompleteMessage
      })
    } catch (err) {
      notifyError(err, t.boot.failure.signInFailed)
    } finally {
      setBusy(null)
    }
  }

  const openLogs = () => void window.hermesDesktop?.revealLogs().catch(() => undefined)
  const copy = t.boot.failure

  const label = signInLabel(remoteReauth, {
    identityProvider: copy.identityProvider,
    remoteGateway: copy.signInToRemoteGateway,
    withProvider: copy.signInWithProvider
  })

  const content =
    !startup || startup.kind === 'setup' ? (
      <GatewaySetupPanel onConfigured={() => window.location.reload()} />
    ) : (
      <GatewayRecoveryPanel
        busy={busy}
        diagnostic={startup.diagnostic}
        kind={startup.kind}
        logs={logs}
        onOpenLogs={openLogs}
        onRetry={() => void retry()}
        onSettings={openGatewaySettings}
        onSignIn={() => void signInRemote()}
        remoteUrl={startup.remoteUrl}
        showLogs={showLogs}
        signInLabel={label}
        toggleLogs={() => setShowLogs(v => !v)}
      />
    )

  return (
    <div className="fixed inset-0 z-[1400] flex items-center justify-center bg-(--ui-chat-surface-background) p-6">
      <div className="w-full max-w-[43rem] overflow-hidden rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) p-5 shadow-nous">
        {content}
      </div>
    </div>
  )
}
