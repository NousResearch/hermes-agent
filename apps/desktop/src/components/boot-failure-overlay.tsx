import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { AlertTriangle, FileText, Loader2, RefreshCw, Wrench } from '@/lib/icons'
import { $desktopBoot } from '@/store/boot'
import { $desktopOnboarding } from '@/store/onboarding'

type BusyAction = 'local' | 'repair' | 'retry' | null

// Recovery surface for a hard boot failure (gateway never came up, backend
// exited during startup, bootstrap latched, …). Without this the app shell
// renders dead — "gateway offline", no composer, only a toast — with no way
// to retry, repair the install, switch the gateway, or find the logs.
export function BootFailureOverlay() {
  const boot = useStore($desktopBoot)
  const onboarding = useStore($desktopOnboarding)
  const [busy, setBusy] = useState<BusyAction>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [showLogs, setShowLogs] = useState(false)

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
  }, [visible])

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
    // applyConnectionConfig reloads the window from the main process.
    await window.hermesDesktop?.applyConnectionConfig({ mode: 'local' }).catch(() => undefined)
    setBusy(null)
  }

  const openLogs = () => void window.hermesDesktop?.revealLogs().catch(() => undefined)

  return (
    <div className="fixed inset-0 z-[1400] flex items-center justify-center bg-(--ui-chat-surface-background) p-6">
      <div className="w-full max-w-[40rem] overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) shadow-sm">
        <div className="flex items-start gap-3 border-b border-(--ui-stroke-tertiary) px-5 py-4">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-destructive/10 text-destructive">
            <AlertTriangle className="size-5" />
          </div>
          <div>
            <h2 className="text-[0.9375rem] font-semibold tracking-tight">Hermes couldn't start</h2>
            <p className="mt-1 text-[0.8125rem] leading-5 text-(--ui-text-tertiary)">
              The background gateway didn't come up. Try one of the recovery steps below — nothing here deletes your
              chats or settings.
            </p>
          </div>
        </div>

        <div className="grid gap-4 p-5">
          <div className="rounded-2xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-xs text-destructive">
            {boot.error}
          </div>

          <div className="grid gap-2">
            <div className="flex flex-wrap gap-2">
              <Button disabled={Boolean(busy)} onClick={() => void retry()}>
                {busy === 'retry' ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
                Retry
              </Button>
              <Button disabled={Boolean(busy)} onClick={() => void repair()} variant="outline">
                {busy === 'repair' ? <Loader2 className="size-4 animate-spin" /> : <Wrench className="size-4" />}
                Repair install
              </Button>
              <Button disabled={Boolean(busy)} onClick={() => void switchToLocalGateway()} variant="outline">
                {busy === 'local' ? <Loader2 className="size-4 animate-spin" /> : null}
                Use local gateway
              </Button>
              <Button onClick={openLogs} variant="ghost">
                <FileText className="size-4" />
                Open logs
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Repair re-runs the installer and can take a few minutes on a fresh machine.
            </p>
          </div>

          {logs.length > 0 ? (
            <div className="grid gap-2">
              <button
                className="self-start text-xs font-medium text-muted-foreground transition hover:text-foreground"
                onClick={() => setShowLogs(v => !v)}
                type="button"
              >
                {showLogs ? 'Hide' : 'Show'} recent logs
              </button>
              {showLogs ? (
                <pre className="max-h-48 overflow-auto rounded-2xl border border-border bg-secondary/30 p-3 font-mono text-[0.7rem] leading-4 text-muted-foreground">
                  {logs.slice(-40).join('')}
                </pre>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
