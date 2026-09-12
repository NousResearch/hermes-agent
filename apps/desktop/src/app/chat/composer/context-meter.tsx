import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { getGlobalModelInfo } from '@/api/models'
import { useSessionView } from '@/app/chat/session-view'
import { ContextUsagePanel } from '@/app/shell/context-usage-panel'
import { useContextBreakdown } from '@/app/shell/hooks/use-context-breakdown'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { compactNumber } from '@/lib/format'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $gateway } from '@/store/gateway'
import { $currentUsage, $sessions, sessionMatchesStoredId } from '@/store/session'
import { ambientRequestFor } from '@/store/session-gone-latch'
import type { UsageStats } from '@/types/hermes'

/** Composer context meter (primary pane only): live % + mini bar beside the
 *  model pill; the popover adds in/out/total, cache hit and cost. Tiles keep
 *  the status quo — their gateway ownership differs, a follow-up. */
export function ComposerContextMeter() {
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const busy = useStore(view.$awaitingResponse)
  const viewModel = useStore(view.$model)
  const viewProvider = useStore(view.$provider)
  const gateway = useStore($gateway)
  const currentUsage = useStore($currentUsage)
  const [open, setOpen] = useState(false)

  const requestGateway = useMemo(
    () =>
      gateway
        ? ambientRequestFor(gateway)
        : async <T,>(): Promise<T> => {
            throw new Error('no gateway')
          },
    [gateway]
  )

  // Draft (no session yet): show the SELECTED model's window at 0% so an
  // empty chat still answers "how much room do I have". Never show another
  // model's window — a mismatch hides the meter instead of lying.
  const [draftInfo, setDraftInfo] = useState<{ max: number; model: string; provider: string } | null>(null)

  useEffect(() => {
    if (runtimeId) {
      return
    }

    let cancelled = false

    // The desktop bridge is absent in unit tests (and any non-app host) —
    // never let a missing bridge fail the render.
    try {
      getGlobalModelInfo()
        .then(info => {
          if (!cancelled) {
            const max = info.effective_context_length || info.auto_context_length || null

            setDraftInfo(max ? { max, model: info.model, provider: info.provider } : null)
          }
        })
        .catch(() => undefined)
    } catch {
      /* no bridge */
    }

    return () => {
      cancelled = true
    }
  }, [view, runtimeId])

  // Same read as the statusbar gauge: measured occupancy once a turn has run,
  // estimate from the live prompt + transcript before that. Keyed to the live
  // runtime session (not the stored id, which doesn't exist yet for a fresh chat).
  // `persist` hands the hook the stored row's scope + counters so a cold boot
  // can paint the durable last-known read while the live agent rebinds.
  const storedRow = useStoreSelector($sessions, sessions =>
    storedId ? (sessions.find(session => sessionMatchesStoredId(session, storedId)) ?? null) : null
  )
  const persist = useMemo(
    () =>
      runtimeId && storedRow
        ? {
            scope: {
              connectionId: storedRow.connection_id ?? '',
              profile: storedRow.profile ?? 'default'
            },
            storedSessionId: storedId ?? '',
            version: {
              input_tokens: storedRow.input_tokens,
              message_count: storedRow.message_count,
              model: storedRow.model,
              output_tokens: storedRow.output_tokens
            }
          }
        : null,
    [runtimeId, storedId, storedRow]
  )
  const { breakdown, loading } = useContextBreakdown({
    busy,
    enabled: Boolean(gateway && runtimeId),
    persist,
    refreshKey: viewModel,
    requestGateway,
    sessionId: runtimeId
  })

  const usage: UsageStats = useMemo(
    () =>
      breakdown
        ? {
            ...currentUsage,
            context_estimated: breakdown.context_estimated,
            context_max: breakdown.context_max,
            context_percent: breakdown.context_percent,
            context_used: breakdown.context_used
          }
        : currentUsage,
    [breakdown, currentUsage]
  )

  if (!runtimeId) {
    const windowMax = draftInfo?.max

    return (
      <DropdownMenu onOpenChange={setOpen} open={open}>
        <Tip
          label={
            windowMax
              ? `Empty chat · ${viewModel} · window ${compactNumber(windowMax)}`
              : `Empty chat · ${viewModel}`
          }
          side="top"
        >
          <DropdownMenuTrigger asChild>
            <Button
              aria-label={windowMax ? `Empty chat, context window ${compactNumber(windowMax)}` : `Empty chat · ${viewModel}`}
              className="h-(--composer-control-size) min-w-0 shrink gap-1.5 rounded-md px-2 text-xs font-normal tabular-nums text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
              type="button"
              variant="ghost"
            >
              <span className="flex items-center text-(--ui-text-tertiary)">
                <svg aria-hidden="true" height="20" viewBox="0 0 20 20" width="20">
                  <circle className="stroke-(--ui-bg-tertiary)" cx="10" cy="10" fill="none" r={7} strokeWidth="2.5" />
                </svg>
              </span>
              <span>0%</span>
            </Button>
          </DropdownMenuTrigger>
        </Tip>
        <DropdownMenuContent align="end" className="w-64 p-3 text-[0.6875rem] text-muted-foreground" side="top" sideOffset={8}>
          <div className="flex flex-col gap-1">
            <div className="truncate font-mono text-foreground">{viewModel}</div>
            <div className="flex items-center justify-between gap-2">
              <span>{viewProvider}</span>
              {windowMax && <span className="shrink-0 tabular-nums text-foreground">Window {compactNumber(windowMax)}</span>}
            </div>
          </div>
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }

  const percent = Math.max(0, Math.min(100, Math.round(usage.context_percent ?? 0)))
  const hit = usage.cache_hit_pct
  const cost = usage.cost_usd ?? 0
  const restored = breakdown?.context_source === 'restored'

  // Ring color follows fullness: calm → warming → hot.
  const ringClass = percent >= 85 ? 'text-red-500' : percent >= 60 ? 'text-amber-500' : 'text-emerald-500'
  const radius = 7
  const circumference = 2 * Math.PI * radius

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <Tip
        label={
          restored
            ? `Restored context ${percent}% (~${compactNumber(usage.context_used ?? 0)} of ${compactNumber(usage.context_max ?? 0)}) — refreshing`
            : `Context usage ${percent}% (${compactNumber(usage.context_used ?? 0)} of ${compactNumber(usage.context_max ?? 0)})`
        }
        side="top"
      >
        <DropdownMenuTrigger asChild>
            <Button
              aria-label={restored ? `Restored context usage ${percent} percent` : `Context usage ${percent} percent`}
            className="h-(--composer-control-size) min-w-0 shrink gap-1.5 rounded-md px-2 text-xs font-normal tabular-nums text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
            type="button"
            variant="ghost"
          >
            <span className={`flex items-center ${ringClass}`}>
              <svg aria-hidden="true" height="20" viewBox="0 0 20 20" width="20">
                <circle
                  className="stroke-(--ui-bg-tertiary)"
                  cx="10"
                  cy="10"
                  fill="none"
                  r={radius}
                  strokeWidth="2.5"
                />
                <circle
                  cx="10"
                  cy="10"
                  fill="none"
                  r={radius}
                  stroke="currentColor"
                  strokeDasharray={circumference}
                  strokeDashoffset={circumference * (1 - percent / 100)}
                  strokeLinecap="round"
                  strokeWidth="2.5"
                  transform="rotate(-90 10 10)"
                />
              </svg>
            </span>
            <span className="tabular-nums">{percent}%</span>
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="w-72 p-0" side="top" sideOffset={8}>
        <ContextUsagePanel breakdown={breakdown} loading={loading} usage={usage} />
        <div className="flex flex-col gap-1 border-t border-(--ui-stroke-tertiary)/40 px-3 py-2 text-[0.6875rem] text-muted-foreground">
          <div className="flex items-center justify-between gap-2">
            <span>
              In {compactNumber(usage.input)} · Out {compactNumber(usage.output)}
            </span>
            <span className="shrink-0 tabular-nums text-foreground">Total {compactNumber(usage.total)}</span>
          </div>
          <div className="flex items-center justify-between gap-2">
            <span>Cache hit {hit != null ? `${Math.round(hit)}%` : '—'}</span>
            <span className="shrink-0 tabular-nums">
              {usage.calls} calls{cost >= 0.01 ? ` · $${cost.toFixed(2)}` : ''}
            </span>
          </div>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
