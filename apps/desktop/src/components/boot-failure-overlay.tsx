import { useStore } from '@nanostores/react'
import { type ComponentProps, lazy, type ReactNode, Suspense, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorIcon } from '@/components/ui/error-state'
import { Loader } from '@/components/ui/loader'
import { LogView } from '@/components/ui/log-view'
import type { DesktopConnectionConfig } from '@/global'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { ChevronLeft, ExternalLink, FileText, Loader2, LogIn, RefreshCw, SlidersHorizontal, Wrench } from '@/lib/icons'
import { $desktopBoot } from '@/store/boot'
import { notify, notifyError } from '@/store/notifications'
import { $desktopOnboarding } from '@/store/onboarding'

import type { RemoteReauth } from './boot-failure-reauth'
import {
  deriveProviderShape,
  isRemoteConfig,
  isRemoteReauthFailure,
  signInLabel,
  sshFailureMessage
} from './boot-failure-reauth'

// The recovery "Gateway settings" view embeds the real Settings → Gateway panel
// (identical URL/auth/test/save controls — no parallel form to drift). Lazy so
// it stays out of the always-mounted overlay's bundle until opened.
const GatewaySettings = lazy(() =>
  import('@/app/settings/gateway-settings').then(module => ({ default: module.GatewaySettings }))
)

type BusyAction = 'local' | 'repair' | 'retry' | 'signin' | null
type RecoveryView = 'connect' | 'recovery'

// A remote gateway whose access cookie has lapsed (e.g. the dashboard
// restarted on the remote box) boots into this overlay with a reauth-shaped
// error. The local-recovery buttons (Retry resets the local bootstrap latch;
// Repair re-runs the installer) are no-ops for that case — the only fix is to
// re-establish the remote session. The detection + copy helpers live in
// ./boot-failure-reauth so they're unit-testable without a React render.

// Recovery surface for a hard boot failure (gateway never came up, backend
// exited during startup, bootstrap latched, …). Without this the app shell
// renders dead — "gateway offline", no composer, only a toast — with no way
// to retry, repair the install, switch the gateway, or find the logs.
export function BootFailureOverlay() {
  const boot = useStore($desktopBoot)
  const onboarding = useStore($desktopOnboarding)
  const { t } = useI18n()
  const [busy, setBusy] = useState<BusyAction>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [showLogs, setShowLogs] = useState(false)
  const [remoteReauth, setRemoteReauth] = useState<RemoteReauth | null>(null)
  const [connectionConfig, setConnectionConfig] = useState<DesktopConnectionConfig | null>(null)
  // A remote/cloud backend that failed to boot is fixable from gateway settings,
  // so the escape hatch earns emphasis (local failures keep it as a quiet ghost).
  const [remoteFailure, setRemoteFailure] = useState(false)
  // Swap the card body to the embedded Gateway settings panel in place of routing
  // to the full Settings page (keeps the user on the recovery surface, no z-index
  // juggling, no second connection form to maintain).
  const [view, setView] = useState<RecoveryView>('recovery')

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

  // Resolve whether this boot failure is a remote-gateway reauth so we can
  // offer the actionable "Sign in" path instead of the local-only recovery
  // buttons. Runs whenever the overlay becomes visible.
  useEffect(() => {
    if (!visible) {
      setRemoteReauth(null)
      setConnectionConfig(null)
      setRemoteFailure(false)
      setView('recovery')

      return
    }

    let cancelled = false

    void (async () => {
      const desktop = window.hermesDesktop

      if (!desktop?.getConnectionConfig) {
        return
      }

      let config: DesktopConnectionConfig

      try {
        config = await desktop.getConnectionConfig()
      } catch {
        return
      }

      if (cancelled) {
        return
      }

      setConnectionConfig(config)
      setRemoteFailure(isRemoteConfig(config))

      if (!isRemoteReauthFailure(config, boot.error)) {
        return
      }

      // Best-effort probe for the provider shape so the button copy matches
      // what the user will see in the login window (password form vs OAuth
      // redirect). Probe failure just keeps the generic copy.
      let shape = deriveProviderShape(null)

      try {
        const probe = await desktop.probeConnectionConfig(config.remoteUrl)
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
    await window.hermesDesktop?.resetBootstrap().catch(() => undefined)
    window.location.reload()
  }

  const repair = async () => {
    setBusy('repair')
    await window.hermesDesktop?.repairBootstrap().catch(() => undefined)
    window.location.reload()
  }

  const switchToLocalGateway = async () => {
    setBusy('local')
    // Soft apply: tears down the primary and re-dials in place (shell stays).
    await window.hermesDesktop?.applyConnectionConfig({ mode: 'local' }).catch(() => undefined)
    setBusy(null)
  }

  // Clear this gateway's stale auth first, then re-establish it through the
  // connection's owning login flow. Hermes Cloud must reuse its portal session
  // and per-agent cascade; generic remote gateways use native/embedded OAuth.
  // Reload after success so boot mints a fresh ticket against the new session.
  const signInRemote = async () => {
    if (!remoteReauth) {
      return
    }

    setBusy('signin')

    try {
      const desktop = window.hermesDesktop

      await desktop?.oauthLogoutConnectionConfig?.(remoteReauth.url)

      let result: { connected?: boolean } | undefined

      if (connectionConfig?.mode === 'cloud' && desktop?.cloud) {
        const status = await desktop.cloud.status()

        if (!status.signedIn) {
          const login = await desktop.cloud.login()

          if (!login.signedIn) {
            notify({
              kind: 'warning',
              title: t.boot.failure.signInIncompleteTitle,
              message: t.boot.failure.signInIncompleteMessage
            })

            return
          }
        }

        result = await desktop.cloud.agentSignIn(remoteReauth.url)
      } else {
        result = await desktop?.oauthLoginConnectionConfig(remoteReauth.url)
      }

      if (result?.connected) {
        if (connectionConfig?.mode === 'cloud') {
          await desktop?.resetBootstrap().catch(() => undefined)
        }

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

  // Recovery actions are shaped by the failure kind so the leading (primary)
  // button is the one that actually fixes it: Sign in for a lapsed remote
  // session, Connection settings for any other remote failure (local Retry /
  // Repair can't revive a dead remote — Repair is dropped there), Retry for a
  // local backend. Open logs is always appended.
  type RecoveryVariant = ComponentProps<typeof Button>['variant']
  interface RecoveryAction {
    key: string
    label: string
    onClick: () => void
    icon?: ReactNode
    variant?: RecoveryVariant
    busy?: Exclude<BusyAction, null>
  }

  const settingsAction: RecoveryAction = {
    key: 'settings',
    label: copy.gatewaySettings,
    onClick: () => setView('connect'),
    icon: <SlidersHorizontal />
  }

  const retryAction: RecoveryAction = {
    key: 'retry',
    label: copy.retry,
    onClick: () => void retry(),
    icon: <RefreshCw />,
    busy: 'retry'
  }

  const localAction: RecoveryAction = {
    key: 'local',
    label: copy.useLocalGateway,
    onClick: () => void switchToLocalGateway(),
    variant: 'secondary',
    busy: 'local'
  }

  let actions: RecoveryAction[]
  let hint: string
  // The electron boot path flags a Nous Cloud backend-down (502/503/504) with
  // the structured isCloudBackendDown/statusCode it carries through boot
  // progress. When set, the recovery screen leads with the cloud-specific
  // guidance instead of the generic remote-failure copy (#85335).
  const cloudDown = Boolean(boot.isCloudBackendDown)

  if (remoteReauth) {
    actions = [
      {
        key: 'signin',
        label: copy.signOutAndSignIn,
        onClick: () => void signInRemote(),
        icon: <LogIn />,
        busy: 'signin'
      },
      { ...settingsAction, variant: 'secondary' },
      localAction
    ]
    hint = copy.remoteSignInHint(label)
  } else if (cloudDown) {
    // A Nous Cloud agent is down — the user cannot restart the managed
    // instance and Repair is local-only. Lead with the paths that actually
    // resolve it: check the portal (status/instance controls), switch to the
    // local gateway, retry, or get support on Discord. Portal/Discord are
    // buttons (not URLs buried in the hint prose) so localized hints can't
    // drift the links.
    actions = [
      {
        key: 'portal',
        label: copy.cloudDownCheckPortal,
        onClick: () => openExternalLink('https://portal.nousresearch.com'),
        icon: <ExternalLink />
      },
      localAction,
      { ...retryAction, variant: 'secondary' },
      {
        key: 'discord',
        label: copy.cloudDownDiscord,
        onClick: () => openExternalLink('https://discord.gg/NousResearch'),
        variant: 'ghost'
      },
      { ...settingsAction, variant: 'ghost' }
    ]
    hint = copy.cloudDownHint
  } else if (remoteFailure) {
    actions = [settingsAction, { ...retryAction, variant: 'secondary' }, localAction]
    hint = copy.remoteFailureHint
  } else {
    // Local failure: Use-local is redundant with Retry (both re-target local), so
    // it's dropped here; keep it for remote failures where it's the fall-back.
    actions = [
      retryAction,
      {
        key: 'repair',
        label: copy.repairInstall,
        onClick: () => void repair(),
        icon: <Wrench />,
        variant: 'secondary',
        busy: 'repair'
      },
      { ...settingsAction, variant: 'ghost' }
    ]
    hint = copy.repairHint
  }

  if (view === 'connect') {
    return (
      <div
        className="fixed inset-0 z-(--z-setup) flex items-center justify-center bg-(--ui-chat-surface-background) p-6"
        // Masks the whole app on boot failure — must stay filled under window
        // glass. Contract: `[data-glass-opaque]` in styles.css.
        data-glass-opaque=""
      >
        <div className="flex max-h-[86vh] w-full max-w-[46rem] flex-col overflow-hidden rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) shadow-nous">
          {/* Subtle back affordance (projects/overlay idiom): muted → foreground
              on hover, no divider. */}
          <button
            className="flex w-full items-center gap-1.5 px-4 pt-4 text-left text-xs text-muted-foreground transition-colors hover:text-foreground"
            onClick={() => setView('recovery')}
            type="button"
          >
            <ChevronLeft className="size-3.5" />
            {copy.back}
          </button>
          <div className="min-h-0 flex-1 pt-4">
            <Suspense fallback={<Loader className="mx-auto my-16 size-6 text-(--ui-text-tertiary)" />}>
              <GatewaySettings embedded />
            </Suspense>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className="fixed inset-0 z-(--z-setup) flex items-center justify-center bg-(--ui-chat-surface-background) p-6"
      // Masks the whole app on boot failure — must stay filled under window
      // glass. Contract: `[data-glass-opaque]` in styles.css.
      data-glass-opaque=""
    >
      <div className="w-full max-w-[40rem] overflow-hidden rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) shadow-nous">
        <div className="flex items-start gap-3 px-5 py-4">
          <ErrorIcon className="mt-0.5" size="1.25rem" />
          <div>
            <h2 className="text-[0.9375rem] font-semibold tracking-tight">
              {remoteReauth ? copy.remoteTitle : cloudDown ? copy.cloudDownTitle : copy.title}
            </h2>
            <p className="mt-1 text-[0.8125rem] leading-5 text-(--ui-text-tertiary)">
              {remoteReauth ? copy.remoteDescription : cloudDown ? copy.cloudDownDescription : copy.description}
            </p>
          </div>
        </div>

        <div className="grid gap-4 p-5 pt-0">
          <div className="rounded-2xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-xs text-destructive">
            {sshFailureMessage(connectionConfig, boot.error, t.settings.gateway)}
          </div>

          <div className="grid gap-2">
            <div className="flex flex-wrap gap-2">
              {actions.map(action => (
                <Button disabled={Boolean(busy)} key={action.key} onClick={action.onClick} variant={action.variant}>
                  {action.busy && busy === action.busy ? <Loader2 className="animate-spin" /> : action.icon}
                  {action.label}
                </Button>
              ))}
              <Button onClick={openLogs} variant="ghost">
                <FileText />
                {copy.openLogs}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">{hint}</p>
          </div>

          {logs.length > 0 ? (
            <div className="grid gap-2">
              <Button
                className="-ml-2 self-start font-medium"
                onClick={() => setShowLogs(v => !v)}
                size="xs"
                type="button"
                variant="text"
              >
                {showLogs ? copy.hideRecentLogs : copy.showRecentLogs}
              </Button>
              {showLogs ? <LogView className="max-h-48">{logs.slice(-40).join('')}</LogView> : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
