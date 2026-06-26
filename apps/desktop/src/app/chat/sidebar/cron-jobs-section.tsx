import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import { Tip } from '@/components/ui/tooltip'
import { getCronJobRuns, type SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $selectedStoredSessionId } from '@/store/session'
import type { CronJob } from '@/types/hermes'

import { jobState, jobTitle, STATE_DOT } from '../../cron/job-state'
import { SidebarPanelLabel } from '../../shell/sidebar-label'

import { SidebarLoadMoreRow } from './load-more-row'

const INACTIVE_STATES = new Set(['completed', 'disabled', 'error', 'paused'])

// Recent runs shown in the inline quick-peek — enough to glance at history
// without turning the sidebar into the full Cron page.
const PEEK_RUN_LIMIT = 5

// Runs are written by the background scheduler tick (no UI signal), so poll the
// open peek so a freshly-fired run shows up within a few seconds.
const PEEK_POLL_INTERVAL_MS = 8000

// Keep the section compact: show a few jobs up front, reveal more in larger
// steps on demand (mirrors the messaging sections in the sidebar).
const INITIAL_VISIBLE_JOBS = 3
const LOAD_MORE_STEP = 10

const relativeFmt = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto', style: 'short' })

// Localized "in 5 min" / "2 hr ago" without hand-rolled strings — picks the
// coarsest sensible unit so a daily job reads "in 14 hr", not "in 840 min".
function relativeTime(targetMs: number, nowMs: number): string {
  const diff = targetMs - nowMs
  const abs = Math.abs(diff)
  const sign = diff < 0 ? -1 : 1

  if (abs < 60_000) {
    return relativeFmt.format(sign * Math.round(abs / 1000), 'second')
  }

  if (abs < 3_600_000) {
    return relativeFmt.format(sign * Math.round(abs / 60_000), 'minute')
  }

  if (abs < 86_400_000) {
    return relativeFmt.format(sign * Math.round(abs / 3_600_000), 'hour')
  }

  return relativeFmt.format(sign * Math.round(abs / 86_400_000), 'day')
}

function nextRunMs(job: CronJob): null | number {
  if (!job.next_run_at) {
    return null
  }

  const ms = Date.parse(job.next_run_at)

  return Number.isNaN(ms) ? null : ms
}

// Runs all belong to the same job, so the run name just repeats the job name —
// the timestamp is what tells them apart. Compact (no year, no seconds) for the
// narrow sidebar.
function formatRunTime(seconds?: null | number): string {
  if (!seconds) {
    return '—'
  }

  const date = new Date(seconds * 1000)

  return Number.isNaN(date.valueOf())
    ? '—'
    : date.toLocaleString(undefined, { day: 'numeric', hour: 'numeric', minute: '2-digit', month: 'short' })
}

interface SidebarCronJobsSectionProps {
  jobs: CronJob[]
  label: string
  max?: number
  // Open a run session's chat (1 click to output).
  onOpenRun: (sessionId: string) => void
  // Open the full Cron page focused on this job (manage / full history).
  onManageJob: (jobId: string) => void
  // Fire the job now.
  onTriggerJob: (jobId: string) => void
  onToggle: () => void
  open: boolean
}

export function SidebarCronJobsSection({
  jobs,
  label,
  max = 50,
  onManageJob,
  onOpenRun,
  onTriggerJob,
  onToggle,
  open
}: SidebarCronJobsSectionProps) {
  const [nowMs, setNowMs] = useState(() => Date.now())
  // Single-open inline peek so the section stays scannable.
  const [peekJobId, setPeekJobId] = useState<null | string>(null)
  // Rows revealed so far; starts compact, grows in steps via "load more".
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_JOBS)

  // One clock for the whole section (rows are pure) so the countdowns tick
  // without re-rendering the rest of the sidebar. Only runs while expanded.
  useEffect(() => {
    if (!open) {
      return
    }

    const id = window.setInterval(() => setNowMs(Date.now()), 1000)

    return () => window.clearInterval(id)
  }, [open])

  // Upcoming first (soonest next run), jobs with no next run sink to the bottom,
  // then alphabetical for stability.
  const sorted = useMemo(() => {
    return [...jobs].sort((a, b) => {
      const an = nextRunMs(a)
      const bn = nextRunMs(b)

      if (an !== null && bn !== null && an !== bn) {
        return an - bn
      }

      if (an === null && bn !== null) {
        return 1
      }

      if (an !== null && bn === null) {
        return -1
      }

      return jobTitle(a).localeCompare(jobTitle(b))
    })
  }, [jobs])

  const cap = Math.min(visibleCount, max)
  const shown = sorted.slice(0, cap)
  const hiddenCount = Math.min(sorted.length, max) - shown.length
  // When capped, signal "50+" rather than implying the list is complete.
  const countLabel = jobs.length > max ? `${max}+` : String(jobs.length)

  return (
    <SidebarGroup className="shrink-0 p-0 pb-1">
      <div className="group/section flex shrink-0 items-center justify-between pb-1 pt-1.5">
        <button
          className="group/section-label flex w-fit items-center gap-1 bg-transparent text-left leading-none"
          onClick={onToggle}
          type="button"
        >
          <SidebarPanelLabel>{label}</SidebarPanelLabel>
          <span className="text-[0.6875rem] font-medium text-(--ui-text-quaternary)">{countLabel}</span>
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
            open={open}
          />
        </button>
      </div>
      {open && (
        <SidebarGroupContent className="flex max-h-72 flex-col gap-px overflow-x-hidden overflow-y-auto overscroll-contain pb-1.75 compact:max-h-none compact:overflow-visible">
          {shown.map(job => (
            <CronJobSidebarRow
              expanded={peekJobId === job.id}
              job={job}
              key={job.id}
              nowMs={nowMs}
              onManage={() => onManageJob(job.id)}
              onOpenRun={onOpenRun}
              onTogglePeek={() => setPeekJobId(prev => (prev === job.id ? null : job.id))}
              onTrigger={() => onTriggerJob(job.id)}
            />
          ))}
          {hiddenCount > 0 && (
            <SidebarLoadMoreRow
              onClick={() => setVisibleCount(count => count + LOAD_MORE_STEP)}
              step={Math.min(LOAD_MORE_STEP, hiddenCount)}
            />
          )}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

function CronJobSidebarRow({
  expanded,
  job,
  nowMs,
  onManage,
  onOpenRun,
  onTogglePeek,
  onTrigger
}: {
  expanded: boolean
  job: CronJob
  nowMs: number
  onManage: () => void
  onOpenRun: (sessionId: string) => void
  onTogglePeek: () => void
  onTrigger: () => void
}) {
  const { t } = useI18n()
  const c = t.cron
  const state = jobState(job)
  const next = nextRunMs(job)
  const label = jobTitle(job)

  const meta = INACTIVE_STATES.has(state) ? (c.states[state] ?? state) : next !== null ? relativeTime(next, nowMs) : '—'

  return (
    <div>
      <div className="group/cron relative grid min-h-[1.625rem] grid-cols-[minmax(0,1fr)_auto] items-center rounded-md hover:bg-(--chrome-action-hover)">
        {/* Lead with the dot in the same w-3.5 cell + pl-2 the session rows use
            so the cron dots line up with the sessions above; the caret sits next
            to the label (matching the other sidebar disclosures) and the whole
            label area toggles the run peek. */}
        <button
          aria-expanded={expanded}
          aria-label={expanded ? c.hideRuns : c.showRuns}
          className="flex min-w-0 items-center gap-1.5 bg-transparent py-0.5 pl-2 pr-1 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
          onClick={onTogglePeek}
          title={label}
          type="button"
        >
          <span className="grid w-3.5 shrink-0 place-items-center">
            <span
              aria-hidden="true"
              className={cn(
                'size-1 rounded-full',
                STATE_DOT[state] ?? 'bg-(--ui-text-quaternary)',
                state === 'running' && 'size-1.5 animate-pulse'
              )}
            />
          </span>
          <span className="min-w-0 truncate text-[0.8125rem] text-(--ui-text-secondary) group-hover/cron:text-foreground">
            {label}
          </span>
          <DisclosureCaret
            className={cn(
              'shrink-0 text-(--ui-text-tertiary) transition',
              expanded ? 'opacity-100' : 'opacity-0 group-hover/cron:opacity-100'
            )}
            open={expanded}
          />
        </button>
        {/* Trailing cluster: countdown by default, quick actions on hover. */}
        <div className="flex items-center gap-0.5 justify-self-end pr-1">
          <span className="text-[0.6875rem] text-(--ui-text-tertiary) tabular-nums group-hover/cron:hidden">
            {meta}
          </span>
          <div className="hidden items-center gap-0.5 group-hover/cron:flex">
            <Tip label={c.triggerNow}>
              <button
                aria-label={c.triggerNow}
                className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
                onClick={onTrigger}
                type="button"
              >
                <Codicon name="zap" size="0.75rem" />
              </button>
            </Tip>
            <Tip label={c.manage}>
              <button
                aria-label={c.manage}
                className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
                onClick={onManage}
                type="button"
              >
                <Codicon name="watch" size="0.75rem" />
              </button>
            </Tip>
          </div>
        </div>
      </div>
      {expanded && <CronJobSidebarRuns jobId={job.id} onOpenRun={onOpenRun} />}
    </div>
  )
}

function CronJobSidebarRuns({ jobId, onOpenRun }: { jobId: string; onOpenRun: (sessionId: string) => void }) {
  const { t } = useI18n()
  const c = t.cron
  const selectedSessionId = useStore($selectedStoredSessionId)
  const [runs, setRuns] = useState<null | SessionInfo[]>(null)

  useEffect(() => {
    let cancelled = false

    const load = () =>
      getCronJobRuns(jobId, PEEK_RUN_LIMIT)
        .then(result => {
          if (!cancelled) {
            setRuns(result)
          }
        })
        .catch(() => {
          if (!cancelled) {
            setRuns(prev => prev ?? [])
          }
        })

    void load()

    const intervalId = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void load()
      }
    }, PEEK_POLL_INTERVAL_MS)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [jobId])

  return (
    <div className="mb-1 ml-[1.375rem] flex flex-col gap-px">
      {runs === null ? (
        <div className="flex items-center gap-1.5 py-1 pl-1 text-[0.6875rem] text-(--ui-text-tertiary)">
          <GlyphSpinner ariaLabel={c.loading} className="text-[0.75rem]" />
        </div>
      ) : runs.length === 0 ? (
        <div className="py-1 pl-1 text-[0.6875rem] text-(--ui-text-tertiary)">{c.noRuns}</div>
      ) : (
        <>
          {runs.map(run => (
            <button
              className={cn(
                'truncate rounded-md px-1.5 py-0.5 text-left text-[0.6875rem] tabular-nums focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40',
                run.id === selectedSessionId
                  ? 'bg-(--ui-row-active-background) text-foreground'
                  : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
              )}
              key={run.id}
              onClick={() => onOpenRun(run.id)}
              type="button"
            >
              {formatRunTime(run.last_active || run.started_at)}
            </button>
          ))}
        </>
      )}
    </div>
  )
}
