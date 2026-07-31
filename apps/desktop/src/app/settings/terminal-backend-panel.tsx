import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { getTerminalBackends, selectTerminalBackend } from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle, Check, Loader2, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { TerminalBackendInfo, TerminalBackendsResponse } from '@/types/hermes'

import { Pill } from './primitives'

interface TerminalBackendPanelProps {
  /** Re-read the parent toolset list after a backend change so any derived
   *  pills stay in sync. */
  onConfiguredChange?: () => void
}

function StatusPill({ backend }: { backend: TerminalBackendInfo }) {
  const { t } = useI18n()
  const copy = t.settings.toolsets.terminalBackend

  if (backend.status === 'ready') {
    return (
      <Pill tone="primary">
        <Check className="size-3" />
        {copy.ready}
      </Pill>
    )
  }

  return (
    <Pill tone="muted">
      <AlertTriangle className="size-3" />
      {backend.status === 'needs_setup' ? copy.needsSetup : copy.unavailable}
    </Pill>
  )
}

/**
 * Terminal execution backend picker — the Capabilities-tab counterpart of the
 * `terminal.backend` config enum. Each backend row carries a live health probe
 * (Docker daemon reachable, SSH host configured, Modal/Daytona credentials
 * present) so users see Ready / Needs-setup guidance instead of a bare
 * dropdown. Selecting a needs-setup backend is allowed — the row shows what's
 * missing rather than blocking, matching the CLI configurator.
 */
export function TerminalBackendPanel({ onConfiguredChange }: TerminalBackendPanelProps) {
  const { t } = useI18n()
  const copy = t.settings.toolsets.terminalBackend
  const [data, setData] = useState<TerminalBackendsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [selecting, setSelecting] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)

    try {
      setData(await getTerminalBackends())
    } catch (err) {
      notifyError(err, copy.failedLoad)
    } finally {
      setLoading(false)
    }
  }, [copy.failedLoad])

  useEffect(() => {
    void refresh()
  }, [refresh])

  async function handleSelect(backend: TerminalBackendInfo) {
    if (backend.active || selecting) {
      return
    }

    setSelecting(backend.name)

    try {
      await selectTerminalBackend(backend.name)
      // Mirror the backend write locally so the active highlight tracks the
      // new selection without a refetch (probes are unchanged by a select).
      setData(current =>
        current
          ? {
              ...current,
              active: backend.name,
              backends: current.backends.map(b => ({ ...b, active: b.name === backend.name }))
            }
          : current
      )
      notify({ kind: 'success', title: copy.selectedTitle, message: copy.selectedMessage(backend.label) })
      onConfiguredChange?.()
    } catch (err) {
      notifyError(err, copy.failedSelect(backend.label))
    } finally {
      setSelecting(null)
    }
  }

  if (loading && !data) {
    return (
      <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        {copy.loading}
      </div>
    )
  }

  if (!data) {
    return null
  }

  return (
    <div className="grid gap-1.5">
      <div className="flex items-baseline justify-between gap-2 px-0.5">
        <span className="text-[0.72rem] font-medium">{copy.sectionTitle}</span>
        <Button disabled={loading} onClick={() => void refresh()} size="sm" variant="text">
          <RefreshCw className={cn('size-3.5', loading && 'animate-spin')} />
        </Button>
      </div>
      <div className="grid gap-1">
        {data.backends.map(backend => (
          <button
            aria-pressed={backend.active}
            className={cn(
              'grid gap-0.5 rounded-lg border px-2.5 py-2 text-left transition',
              backend.active
                ? 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
                : 'border-transparent bg-background/55 hover:bg-accent/40'
            )}
            disabled={selecting !== null}
            key={backend.name}
            onClick={() => void handleSelect(backend)}
            type="button"
          >
            <span className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium">{backend.label}</span>
              <StatusPill backend={backend} />
              {backend.active && (
                <Pill tone="primary">
                  <Check className="size-3" />
                  {copy.inUse}
                </Pill>
              )}
              {selecting === backend.name && <Loader2 className="size-3 animate-spin" />}
            </span>
            <span className="text-[0.68rem] text-muted-foreground">{backend.description}</span>
            {backend.status !== 'ready' && backend.detail && (
              <span className="flex items-start gap-1 text-[0.68rem] text-amber-600 dark:text-amber-300">
                <AlertTriangle className="mt-0.5 size-3 shrink-0" />
                {backend.detail}
                {backend.active && ` ${copy.needsSetupHint}`}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
