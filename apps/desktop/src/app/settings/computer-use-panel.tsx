import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { getActionStatus, getComputerUseStatus, grantComputerUsePermissions } from '@/hermes'
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

function PermissionRow({ granted, label, hint }: { granted: boolean | null; label: string; hint: string }) {
  const tone = granted === true ? 'primary' : 'muted'
  const Icon = granted === true ? Check : granted === false ? X : AlertTriangle

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-background/55 p-2.5">
      <div className="min-w-0">
        <span className="text-sm font-medium">{label}</span>
        <p className="mt-0.5 text-[0.7rem] text-muted-foreground">{hint}</p>
      </div>
      <Pill tone={tone}>
        <Icon className="size-3" />
        {granted === true ? 'Granted' : granted === false ? 'Not granted' : 'Unknown'}
      </Pill>
    </div>
  )
}

/**
 * Computer Use preflight card.
 *
 * Computer Use drives the Mac through cua-driver, whose Accessibility +
 * Screen Recording grants attach to cua-driver's OWN TCC identity
 * (`com.trycua.driver` / the installed CuaDriver.app) — not the Hermes
 * desktop app. So this card reflects the driver's real grant state and
 * triggers a grant via `cua-driver permissions grant`, which launches
 * CuaDriver via LaunchServices so the macOS dialog is attributed correctly.
 *
 * Binary install/upgrade still lives in the cua-driver provider's post-setup
 * runner below this card (the generic ToolsetConfigPanel).
 */
export function ComputerUsePanel({ onConfiguredChange }: ComputerUsePanelProps) {
  const [status, setStatus] = useState<ComputerUseStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [granting, setGranting] = useState(false)
  const activeRef = useRef(false)

  const refresh = useCallback(async () => {
    try {
      const next = await getComputerUseStatus()
      setStatus(next)
    } catch (err) {
      notifyError(err, 'Could not read Computer Use status')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    activeRef.current = true
    void refresh()

    return () => {
      activeRef.current = false
    }
  }, [refresh])

  const grant = useCallback(async () => {
    setGranting(true)

    try {
      const started = await grantComputerUsePermissions()

      if (!started.ok) {
        notifyError(new Error('spawn failed'), 'Could not request permissions')

        return
      }

      notify({
        kind: 'info',
        title: 'Approve in System Settings',
        message: 'macOS will show a permission dialog attributed to CuaDriver. Approve it, then return here.'
      })

      // Poll the grant action until it exits (the driver waits for the user to
      // flip the switch), then re-read the live permission state.
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
        notifyError(err, 'Could not request permissions')
      }
    } finally {
      if (activeRef.current) {
        setGranting(false)
      }
    }
  }, [onConfiguredChange, refresh])

  if (loading) {
    return (
      <div className="mt-3 flex items-center gap-2 px-1 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        Checking Computer Use status…
      </div>
    )
  }

  if (!status) {
    return null
  }

  if (!status.platform_supported) {
    return (
      <p className="mt-3 px-1 text-xs text-muted-foreground">
        Computer Use permissions are managed on macOS. On this platform, enable the cua-driver provider below.
      </p>
    )
  }

  if (!status.installed) {
    return (
      <p className="mt-3 px-1 text-xs text-muted-foreground">
        Install the cua-driver backend below to drive macOS. After installing, grant Accessibility and Screen
        Recording here.
      </p>
    )
  }

  const allGranted = status.accessibility === true && status.screen_recording === true

  return (
    <div className="mt-3 grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2 px-1">
        <div className="min-w-0">
          <p className="text-[0.72rem] text-muted-foreground">
            Grants attach to CuaDriver&apos;s own identity (com.trycua.driver), not Hermes — so the dialog is
            attributed to the process that drives your Mac.
          </p>
          {status.version && <p className="text-[0.68rem] text-muted-foreground/80">{status.version}</p>}
        </div>
        <Button onClick={() => void refresh()} size="sm" variant="text">
          <RefreshCw className="size-3.5" />
          Recheck
        </Button>
      </div>

      <PermissionRow
        granted={status.accessibility}
        hint="Lets cua-driver post clicks, keystrokes, and read the accessibility tree."
        label="Accessibility"
      />
      <PermissionRow
        granted={status.screen_recording}
        hint="Lets cua-driver capture screenshots of app windows."
        label="Screen Recording"
      />

      {status.error && (
        <p className="px-1 text-[0.7rem] text-muted-foreground">
          <AlertTriangle className="mr-1 inline size-3" />
          {status.error}
        </p>
      )}

      {allGranted ? (
        <div className="flex items-center gap-1.5 px-1 text-xs text-muted-foreground">
          <Check className="size-3.5" />
          Computer Use is ready. Ask the agent to capture an app and click around.
        </div>
      ) : (
        <Button disabled={granting} onClick={() => void grant()} size="sm">
          {granting ? <Loader2 className="size-3.5 animate-spin" /> : <ExternalLink className="size-3.5" />}
          {granting ? 'Waiting for approval…' : 'Grant permissions'}
        </Button>
      )}
    </div>
  )
}
