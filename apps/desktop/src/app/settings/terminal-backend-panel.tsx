import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  getTerminalBackends,
  type ProfileScope,
  profileScopeKey,
  selectTerminalBackend
} from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle, Check, Loader2, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { TerminalBackendInfo, TerminalBackendsResponse } from '@/types/hermes'

import { hermesConfigCacheWriter } from '../hooks/use-config-record'

import { setNested } from './helpers'
import { Pill } from './primitives'

function pickerText(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

interface TerminalBackendPanelProps {
  /** Re-read the parent toolset list after a backend change so any derived
   *  pills stay in sync. */
  onConfiguredChange?: () => void
  profile?: ProfileScope
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
export function TerminalBackendPanel({ onConfiguredChange, profile }: TerminalBackendPanelProps) {
  const { t } = useI18n()
  const copy = t.settings.toolsets.terminalBackend
  const [data, setData] = useState<TerminalBackendsResponse | null>(null)
  const [dataScope, setDataScope] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const [pendingSelection, setPendingSelection] = useState<{
    backend: string
    generation: number
    lifecycle: { active: boolean; scopeKey: string }
  } | null>(null)

  const scopeKey = profileScopeKey(profile)

  const lifecycle = useMemo(() => ({ active: true, scopeKey }), [scopeKey])
  const selecting = pendingSelection?.lifecycle === lifecycle ? pendingSelection.backend : null

  const refreshGeneration = useRef(0)
  const selectionGeneration = useRef(0)
  const scopedData = dataScope === scopeKey ? data : null

  const refresh = useCallback(async () => {
    const generation = ++refreshGeneration.current
    const requestedScope = scopeKey
    setLoading(true)

    try {
      const next = await getTerminalBackends(profile)

      if (generation !== refreshGeneration.current || !lifecycle.active) {
        return
      }

      setData(next)
      setDataScope(requestedScope)
    } catch (err) {
      if (generation !== refreshGeneration.current || !lifecycle.active) {
        return
      }

      notifyError(err, copy.failedLoad)
    } finally {
      if (generation === refreshGeneration.current && lifecycle.active) {
        setLoading(false)
      }
    }
  }, [copy.failedLoad, lifecycle, profile, scopeKey])

  useEffect(() => {
    lifecycle.active = true
    void refresh()

    return () => {
      lifecycle.active = false
    }
  }, [lifecycle, refresh])

  async function handleSelect(backend: TerminalBackendInfo) {
    if (backend.active || selecting) {
      return
    }

    const generation = ++selectionGeneration.current
    setPendingSelection({ backend: backend.name, generation, lifecycle })

    try {
      await selectTerminalBackend(backend.name, profile)

      if (generation !== selectionGeneration.current || !lifecycle.active) {
        return
      }

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
      hermesConfigCacheWriter(profile)(current =>
        current ? setNested(current, 'terminal.backend', backend.name) : current
      )
      notify({
        kind: 'success',
        title: copy.selectedTitle,
        message: copy.selectedMessage(pickerText(backend.label, backend.name))
      })
      onConfiguredChange?.()
    } catch (err) {
      if (generation !== selectionGeneration.current || !lifecycle.active) {
        return
      }

      notifyError(err, copy.failedSelect(pickerText(backend.label, backend.name)))
    } finally {
      if (generation === selectionGeneration.current && lifecycle.active) {
        setPendingSelection(current =>
          current?.generation === generation && current.lifecycle === lifecycle ? null : current
        )
      }
    }
  }

  if (loading && !scopedData) {
    return (
      <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        {copy.loading}
      </div>
    )
  }

  if (!scopedData) {
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
        {scopedData.backends.map(backend => (
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
              <span className="text-xs font-medium">{pickerText(backend.label, backend.name)}</span>
              <StatusPill backend={backend} />
              {backend.active && (
                <Pill tone="primary">
                  <Check className="size-3" />
                  {copy.inUse}
                </Pill>
              )}
              {selecting === backend.name && <Loader2 className="size-3 animate-spin" />}
            </span>
            <span className="text-[0.68rem] text-muted-foreground">{pickerText(backend.description)}</span>
            {backend.status !== 'ready' && backend.detail && (
              <span className="flex items-start gap-1 text-[0.68rem] text-amber-600 dark:text-amber-300">
                <AlertTriangle className="mt-0.5 size-3 shrink-0" />
                {pickerText(backend.detail)}
                {backend.active && ` ${copy.needsSetupHint}`}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
