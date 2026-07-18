import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { getActionStatus, getComputerUseStatus, grantComputerUsePermissions } from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle, Check, ExternalLink, Loader2, RefreshCw, X } from '@/lib/icons'
import { upsertDesktopActionTask } from '@/store/activity'
import { notify, notifyError } from '@/store/notifications'
import type { ComputerUseStatus } from '@/types/hermes'

import { Pill } from './primitives'

interface ComputerUsePanelProps {
  /** Re-read the parent toolset list after a permission/install change so the
   *  "Configured / Needs keys" pill stays in sync. */
  onConfiguredChange?: () => void
}

function tone(granted: boolean | null) {
  return granted === true ? 'primary' : 'muted'
}

function GrantIcon({ granted }: { granted: boolean | null }) {
  const Icon = granted === true ? Check : granted === false ? X : AlertTriangle

  return <Icon className="size-3" />
}

function PermissionRow({
  granted,
  label,
  hint,
  statuses
}: {
  granted: boolean | null
  label: string
  hint: string
  statuses: { granted: string; notGranted: string; unknown: string }
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-background/55 p-2.5">
      <div className="min-w-0">
        <span className="text-sm font-medium">{label}</span>
        <p className="mt-0.5 text-[0.7rem] text-muted-foreground">{hint}</p>
      </div>
      <Pill tone={tone(granted)}>
        <GrantIcon granted={granted} />
        {granted === true ? statuses.granted : granted === false ? statuses.notGranted : statuses.unknown}
      </Pill>
    </div>
  )
}

/**
 * Cross-platform Computer Use preflight card.
 *
 * cua-driver runs on macOS, Windows, and Linux, but readiness differs: macOS
 * needs two TCC grants (Accessibility + Screen Recording) that attach to
 * cua-driver's own `com.trycua.driver` identity — not Hermes — and are
 * requested via `cua-driver permissions grant` (dialog attributed to
 * CuaDriver). Windows/Linux have no TCC toggles, so readiness is driver health
 * from `cua-driver doctor`. The backend folds both into one `ready` signal.
 *
 * Binary install/upgrade stays in the cua-driver provider's post-setup runner
 * below this card (the generic ToolsetConfigPanel).
 */
export function ComputerUsePanel({ onConfiguredChange }: ComputerUsePanelProps) {
  const c = useI18n().t.settings.computerUse
  const [status, setStatus] = useState<ComputerUseStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [granting, setGranting] = useState(false)
  const activeRef = useRef(false)

  const refresh = useCallback(async () => {
    try {
      setStatus(await getComputerUseStatus())
    } catch (err) {
      notifyError(err, c.statusLoadFailed)
    } finally {
      setLoading(false)
    }
  }, [c.statusLoadFailed])

  useEffect(() => {
    activeRef.current = true
    void refresh()

    return () => void (activeRef.current = false)
  }, [refresh])

  const grant = useCallback(async () => {
    setGranting(true)

    try {
      const started = await grantComputerUsePermissions()

      if (!started.ok) {
        notifyError(new Error('spawn failed'), c.permissionRequestFailed)

        return
      }

      notify({
        kind: 'info',
        title: c.approveTitle,
        message: c.approveMessage
      })

      // The driver waits for the user to flip the switch — poll until it exits.
      for (let attempt = 0; attempt < 150 && activeRef.current; attempt += 1) {
        await new Promise(resolve => window.setTimeout(resolve, 1500))

        if (!activeRef.current) {
          break
        }

        const polled = await getActionStatus(started.name, 200)
        upsertDesktopActionTask(polled)

        if (!polled.running) {
          break
        }
      }

      if (activeRef.current) {
        await refresh()
        onConfiguredChange?.()
      }
    } catch (err) {
      if (activeRef.current) {
        notifyError(err, c.permissionRequestFailed)
      }
    } finally {
      if (activeRef.current) {
        setGranting(false)
      }
    }
  }, [c.approveMessage, c.approveTitle, c.permissionRequestFailed, onConfiguredChange, refresh])

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        {c.checking}
      </div>
    )
  }

  if (!status) {
    return null
  }

  if (!status.platform_supported) {
    return <p className="px-1 text-xs text-muted-foreground">{c.unsupported(status.platform)}</p>
  }

  if (!status.installed) {
    return (
      <p className="px-1 text-xs text-muted-foreground">
        {c.installDriver}
        {status.can_grant && c.thenGrant}
      </p>
    )
  }

  const failingChecks = status.checks.filter(c => c.status !== 'ok')

  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2 px-1">
        <div className="min-w-0">
          {status.can_grant ? (
            <p className="text-[0.72rem] text-muted-foreground">{c.identityHint}</p>
          ) : (
            <p className="text-[0.72rem] text-muted-foreground">
              {status.platform === 'linux' ? c.linuxNote : c.windowsNote}
            </p>
          )}
          {status.version && <p className="text-[0.68rem] text-muted-foreground/80">{status.version}</p>}
        </div>
        <Button onClick={() => void refresh()} size="sm" variant="text">
          <RefreshCw className="size-3.5" />
          {c.recheck}
        </Button>
      </div>

      {status.can_grant ? (
        <>
          <PermissionRow
            granted={status.accessibility}
            hint={c.accessibilityHint}
            label={c.accessibility}
            statuses={c}
          />
          <PermissionRow
            granted={status.screen_recording}
            hint={c.screenRecordingHint}
            label={c.screenRecording}
            statuses={c}
          />
        </>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-background/55 p-2.5">
          <span className="text-sm font-medium">{c.driverHealth}</span>
          <Pill tone={tone(status.ready)}>
            <GrantIcon granted={status.ready} />
            {status.ready === true ? c.ready : status.ready === false ? c.notReady : c.unknown}
          </Pill>
        </div>
      )}

      {failingChecks.map(c => (
        <p className="px-1 text-[0.7rem] text-muted-foreground" key={c.label}>
          <AlertTriangle className="mr-1 inline size-3" />
          {c.label}: {c.message}
        </p>
      ))}

      {status.error && (
        <p className="px-1 text-[0.7rem] text-muted-foreground">
          <AlertTriangle className="mr-1 inline size-3" />
          {status.error}
        </p>
      )}

      {status.ready ? (
        <div className="flex items-center gap-1.5 px-1 text-xs text-muted-foreground">
          <Check className="size-3.5" />
          {c.readyHint}
        </div>
      ) : (
        status.can_grant && (
          <Button disabled={granting} onClick={() => void grant()} size="sm">
            {granting ? <Loader2 className="size-3.5 animate-spin" /> : <ExternalLink className="size-3.5" />}
            {granting ? c.waitingApproval : c.grantPermissions}
          </Button>
        )
      )}
    </div>
  )
}
