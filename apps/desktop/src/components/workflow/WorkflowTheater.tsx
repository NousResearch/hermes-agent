import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { CheckCircle2, Layers3, Loader2, Sparkles, X, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $caduceus, $caduceusTheaterOpen, closeTheater, openTheater } from '@/store/caduceus'
import { $workflowRun, clearWorkflow, workflowActiveCount } from '@/store/workflow'

import { BudgetGauge, ConcurrencyMeter, Narrator, PhaseLanes, VerifyFeed } from './TheaterPanels'
import { formatElapsed, formatTokens } from './theater-format'

function useTick(active: boolean): number {
  const [, setN] = useState(0)
  useEffect(() => {
    if (!active) {
      return
    }
    const id = window.setInterval(() => setN(n => n + 1), 1000)
    return () => window.clearInterval(id)
  }, [active])
  return Date.now()
}

/**
 * The Orchestration Theater — a live view of a running Caduceus workflow.
 * Self-contained: renders nothing when idle, a floating mini-strip when a
 * workflow is running but collapsed, and a full glassy overlay when expanded.
 */
export function WorkflowTheater() {
  const run = useStore($workflowRun)
  const open = useStore($caduceusTheaterOpen)
  const caduceus = useStore($caduceus)
  const live = !!run && run.status === 'running'
  const now = useTick(live)

  if (!run) {
    return null
  }

  const active = workflowActiveCount(run)
  const total = run.order.length
  const queued = run.order.filter(id => run.agents[id]?.status === 'queued').length
  const elapsedMs = (live ? now : (run.updatedAt ?? now)) - run.startedAt
  const orchestratorModel = caduceus.orchestrator.model

  if (!open) {
    return (
      <button
        className={cn(
          'fixed bottom-14 right-4 z-[1200] flex items-center gap-2 rounded-full border border-(--ui-stroke-tertiary)',
          'bg-(--ui-chat-bubble-background)/90 px-3 py-1.5 text-xs shadow-lg backdrop-blur-md transition-all hover:scale-[1.02]',
          'motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-bottom-2'
        )}
        onClick={openTheater}
        title="Open Orchestration Theater"
        type="button"
      >
        <span className="relative flex size-2">
          {live && (
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-sky-400/70" />
          )}
          <span className={cn('relative inline-flex size-2 rounded-full', live ? 'bg-sky-400' : 'bg-emerald-400')} />
        </span>
        <Sparkles className="size-3.5 text-amber-300" />
        <span className="max-w-[10rem] truncate font-medium text-foreground">{run.name}</span>
        <span className="tabular-nums text-muted-foreground">
          {live ? `${active}/${total}` : `${total} agents`}
        </span>
        <span className="tabular-nums text-muted-foreground/70">{formatTokens(run.tokensOut)} tok</span>
      </button>
    )
  }

  return (
    <div className="fixed inset-0 z-[1250] flex items-end justify-center p-3 sm:items-center">
      <button
        aria-label="Close theater"
        className="absolute inset-0 bg-black/40 backdrop-blur-[2px] motion-safe:animate-in motion-safe:fade-in"
        onClick={closeTheater}
        type="button"
      />
      <div
        className={cn(
          'relative flex h-[78vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-(--ui-stroke-tertiary)',
          'bg-background/95 shadow-2xl backdrop-blur-xl motion-safe:animate-in motion-safe:fade-in motion-safe:zoom-in-95'
        )}
      >
        {/* gradient header rail */}
        <div className="relative border-b border-(--ui-stroke-tertiary) bg-gradient-to-r from-amber-500/10 via-transparent to-sky-500/10 px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="grid size-7 place-items-center rounded-lg bg-amber-400/15 text-amber-300">
              <Sparkles className="size-4" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <h2 className="truncate text-sm font-semibold text-foreground">{run.name}</h2>
                <span className="rounded bg-amber-400/15 px-1.5 py-px text-[0.6rem] font-medium uppercase tracking-wide text-amber-300">
                  ⚕ Caduceus · {caduceus.effort ?? 'high'}
                </span>
              </div>
              {run.description && <p className="truncate text-[0.7rem] text-muted-foreground">{run.description}</p>}
            </div>
            <StatusPill run={run} />
            <span className="flex items-center gap-1 tabular-nums text-xs text-muted-foreground">
              <Loader2 className={cn('size-3', live ? 'animate-spin' : 'opacity-0')} />
              {formatElapsed(elapsedMs)}
            </span>
            <button
              aria-label="Close"
              className="grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
              onClick={closeTheater}
              type="button"
            >
              <X className="size-4" />
            </button>
          </div>
          <div className="mt-2">
            <Narrator lines={run.narrator} />
          </div>
        </div>

        {/* main stage */}
        <div className="flex min-h-0 flex-1 gap-3 px-4 py-3">
          <div className="min-w-0 flex-1">
            <PhaseLanes orchestratorModel={orchestratorModel} run={run} />
          </div>
          {run.verifies.length > 0 && (
            <div className="hidden w-64 shrink-0 overflow-y-auto md:block">
              <VerifyFeed verifies={run.verifies} />
            </div>
          )}
        </div>

        {/* completion banner */}
        {run.status !== 'running' && (
          <div
            className={cn(
              'mx-4 mb-2 flex items-center gap-2 rounded-lg border px-3 py-2 text-xs',
              run.status === 'error'
                ? 'border-rose-500/40 bg-rose-500/10 text-rose-200'
                : 'border-emerald-400/40 bg-emerald-400/10 text-emerald-200'
            )}
          >
            {run.status === 'error' ? <X className="size-4 shrink-0" /> : <CheckCircle2 className="size-4 shrink-0" />}
            <span className="min-w-0 flex-1 truncate">
              {run.status === 'error' ? run.error : run.resultSummary || 'Workflow complete.'}
            </span>
            <button
              className="shrink-0 rounded px-2 py-0.5 text-[0.65rem] text-muted-foreground hover:bg-(--chrome-action-hover) hover:text-foreground"
              onClick={() => {
                clearWorkflow()
                closeTheater()
              }}
              type="button"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* footer meters */}
        <div className="flex items-center gap-4 border-t border-(--ui-stroke-tertiary) px-4 py-2.5">
          <span className="flex shrink-0 items-center gap-1 text-[0.65rem] text-muted-foreground">
            <Layers3 className="size-3" /> {run.agentCount || total} agents
          </span>
          <ConcurrencyMeter active={active} cap={run.concurrencyCap} queued={queued} />
          <BudgetGauge spent={run.budgetSpent || run.tokensOut} total={run.budgetTotal} />
          <span className="flex shrink-0 items-center gap-1 text-[0.65rem] text-muted-foreground">
            <Zap className="size-3 text-amber-300" /> {formatTokens(run.tokensOut)} out
          </span>
        </div>
      </div>
    </div>
  )
}

function StatusPill({ run }: { run: NonNullable<ReturnType<typeof $workflowRun.get>> }) {
  const map = {
    complete: { cls: 'bg-emerald-400/15 text-emerald-300', label: 'complete' },
    error: { cls: 'bg-rose-500/15 text-rose-300', label: 'error' },
    running: { cls: 'bg-sky-400/15 text-sky-300', label: 'running' }
  } as const
  const s = map[run.status]
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.6rem] font-medium uppercase', s.cls)}>
      {run.status === 'running' && <span className="size-1.5 animate-pulse rounded-full bg-current" />}
      {s.label}
    </span>
  )
}
