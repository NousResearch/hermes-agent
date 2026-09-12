import { createCronTriggerController, type CronTriggerController } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { ProfileTag } from '@/app/chat/profile-tag'
import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { ActionsContextMenu, type MenuKit, renderActionItem } from '@/components/ui/actions-menu'
import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import { Tip } from '@/components/ui/tooltip'
import { deleteCronJob, getCronJobRuns, pauseCronJob, resumeCronJob, type SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { fmtDayTime, relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import { confirm } from '@/store/confirm'
import { updateCronJobs } from '@/store/cron'
import { $changeEventsAvailable, $cronChangeTick } from '@/store/live-sync'
import { notify, notifyError } from '@/store/notifications'
import { $selectedStoredSessionId } from '@/store/session'
import type { CronJob } from '@/types/hermes'

import { jobState, jobTitle, STATE_DOT } from '../../cron/job-state'
import { SidebarPanelLabel } from '../../shell/sidebar-label'

import { SidebarRowBody, SidebarRowLabel, SidebarRowLead, SidebarRowShell } from './chrome'
import { SidebarLoadMoreRow } from './load-more-row'

const INACTIVE_STATES = new Set(['completed', 'disabled', 'error', 'paused'])

// Recent runs shown in the inline quick-peek — enough to glance at history
// without turning the sidebar into the full Cron page.
const PEEK_RUN_LIMIT = 5

// Runs are written by the background scheduler tick. cron.changed reloads the
// open peek immediately on event-capable backends (poll drops to a backstop);
// older backends keep the legacy cadence.
const PEEK_POLL_INTERVAL_MS = 8000
const PEEK_BACKSTOP_INTERVAL_MS = 60_000

// Keep the section compact: show a few jobs up front, reveal more in larger
// steps on demand (mirrors the messaging sections in the sidebar).
const INITIAL_VISIBLE_JOBS = 3
const LOAD_MORE_STEP = 10

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

  return Number.isNaN(date.valueOf()) ? '—' : fmtDayTime.format(date)
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
  onTriggerJob: (jobId: string) => Promise<void>
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
  const [triggeringJobIds, setTriggeringJobIds] = useState<ReadonlySet<string>>(() => new Set())
  const triggerControllerRef = useRef<CronTriggerController | null>(null)

  // eslint-disable-next-line no-restricted-syntax -- controller mount identity, not an atom mirror
  useEffect(() => {
    const controller = createCronTriggerController((jobId, running) => {
      if (triggerControllerRef.current !== controller) {
        return
      }

      setTriggeringJobIds(current => {
        const next = new Set(current)

        if (running) {
          next.add(jobId)
        } else {
          next.delete(jobId)
        }

        return next
      })
    })

    triggerControllerRef.current = controller

    return () => {
      triggerControllerRef.current = null
    }
  }, [])

  const triggerJob = (jobId: string) => {
    const controller = triggerControllerRef.current

    if (!controller) {
      return
    }

    void controller.run(jobId, () => onTriggerJob(jobId)).catch(() => undefined)
  }

  const visible = usePaneVisible()

  // One clock for the whole section (rows are pure) so the countdowns tick
  // without re-rendering the rest of the sidebar. Only runs while expanded and visible.
  useEffect(() => {
    if (!open || !visible) {
      return
    }

    const id = window.setInterval(() => setNowMs(Date.now()), 1000)

    return () => window.clearInterval(id)
  }, [open, visible])

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

  return (
    <SidebarGroup className="shrink-0 p-0 pb-1">
      <div className="group/section flex shrink-0 items-center justify-between pb-1 pt-1.5">
        <button
          className="group/section-label flex w-fit min-w-0 items-center gap-1 bg-transparent text-left leading-none"
          onClick={onToggle}
          type="button"
        >
          <SidebarPanelLabel>{label}</SidebarPanelLabel>
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
            open={open}
          />
        </button>
      </div>
      {open && (
        <SidebarGroupContent className="scrollbar-fade flex max-h-72 flex-col gap-px overflow-x-hidden overflow-y-auto overscroll-contain pb-1.75 compact:max-h-none compact:overflow-visible">
          {shown.map(job => (
            <CronJobSidebarRow
              busy={triggeringJobIds.has(job.id)}
              expanded={peekJobId === job.id}
              job={job}
              key={job.id}
              nowMs={nowMs}
              onManage={() => onManageJob(job.id)}
              onOpenRun={onOpenRun}
              onTogglePeek={() => setPeekJobId(prev => (prev === job.id ? null : job.id))}
              onTrigger={() => triggerJob(job.id)}
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
  busy,
  expanded,
  job,
  nowMs,
  onManage,
  onOpenRun,
  onTogglePeek,
  onTrigger
}: {
  busy: boolean
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
  const isPaused = state === 'paused'

  const meta = INACTIVE_STATES.has(state) ? (c.states[state] ?? state) : next !== null ? relativeTime(next, nowMs) : '—'

  // Pause/resume and delete aren't threaded through the sidebar's prop chain, so
  // drive them against the shared $cronJobs atom directly (same path the cron
  // overlay uses) — the sidebar and overlay render from that one atom, so the
  // row updates in place.
  const togglePause = async () => {
    try {
      const updated = isPaused ? await resumeCronJob(job.id) : await pauseCronJob(job.id)
      updateCronJobs(rows => rows.map(row => (row.id === job.id ? updated : row)))
      notify({ kind: 'success', title: isPaused ? c.resumed : c.paused, message: label })
    } catch (err) {
      notifyError(err, c.failedUpdate)
    }
  }

  const remove = async () => {
    const ok = await confirm({
      confirmLabel: t.common.delete,
      description: `${c.deleteDescPrefix}${label}${c.deleteDescSuffix}`,
      destructive: true,
      title: c.deleteTitle
    })

    if (!ok) {
      return
    }

    try {
      await deleteCronJob(job.id)
      updateCronJobs(rows => rows.filter(row => row.id !== job.id))
      notify({ kind: 'success', title: c.deleted, message: label })
    } catch (err) {
      notifyError(err, c.failedDelete)
    }
  }

  // One action set for both the hover buttons and the right-click menu.
  const items = (kit: MenuKit) => (
    <>
      {renderActionItem(kit, { icon: 'zap', key: 'trigger', label: c.triggerNow, onSelect: onTrigger })}
      {renderActionItem(kit, {
        icon: isPaused ? 'play' : 'debug-pause',
        key: 'pause',
        label: isPaused ? c.resume : c.pause,
        onSelect: () => void togglePause()
      })}
      {renderActionItem(kit, { icon: 'watch', key: 'manage', label: c.manage, onSelect: onManage })}
      <kit.Separator />
      {renderActionItem(kit, {
        icon: 'trash',
        key: 'delete',
        label: t.common.delete,
        onSelect: () => void remove(),
        variant: 'destructive'
      })}
    </>
  )

  return (
    <div>
      {/* The shared row chrome, not a copy of it: a cron job and a session sit
          in the same list, so they line up only if one place owns the geometry. */}
      <ActionsContextMenu ariaLabel={c.actionsTitle} contentClassName="w-44" items={items}>
        <SidebarRowShell
          actions={
            /* Trailing cluster: countdown by default, quick actions on hover. */
            <div className="flex items-center gap-0.5">
              <span className="text-[0.6875rem] text-(--ui-text-tertiary) tabular-nums group-hover/cron:hidden">
                {meta}
              </span>
              <div className="hidden items-center gap-0.5 group-hover/cron:flex">
                <Tip label={c.triggerNow}>
                  <button
                    aria-label={c.triggerNow}
                    className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:cursor-wait disabled:opacity-60"
                    disabled={busy}
                    onClick={onTrigger}
                    type="button"
                  >
                    {busy ? (
                      <GlyphSpinner ariaLabel={c.triggerNow} className="text-[0.75rem]" />
                    ) : (
                      <Codicon name="zap" size="0.75rem" />
                    )}
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
          }
          className="group/cron relative hover:bg-(--chrome-action-hover)"
        >
          {/* The caret sits next to the label (matching the other sidebar
              disclosures) and the whole label area toggles the run peek. */}
          <Tip label={label}>
            <SidebarRowBody
              aria-expanded={expanded}
              aria-label={expanded ? c.hideRuns : c.showRuns}
              className="focus-visible:bg-(--chrome-action-hover) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              onClick={onTogglePeek}
            >
              <SidebarRowLead>
                <span
                  aria-hidden="true"
                  className={cn(
                    'size-1 rounded-full',
                    STATE_DOT[state] ?? 'bg-(--ui-text-quaternary)',
                    state === 'running' && 'size-1.5 animate-pulse'
                  )}
                />
              </SidebarRowLead>
              <SidebarRowLabel className="group-hover/cron:text-foreground">{label}</SidebarRowLabel>
              {/* Owning-profile chip — same tag the session rows wear, so a
                  cross-profile list reads as many owners at a glance. */}
              {job.profile ? <ProfileTag profile={job.profile} /> : null}
              <DisclosureCaret
                className={cn(
                  'shrink-0 text-(--ui-text-tertiary) transition',
                  expanded ? 'opacity-100' : 'opacity-0 group-hover/cron:opacity-100'
                )}
                open={expanded}
              />
            </SidebarRowBody>
          </Tip>
        </SidebarRowShell>
      </ActionsContextMenu>
      {expanded && <CronJobSidebarRuns jobId={job.id} onOpenRun={onOpenRun} />}
    </div>
  )
}

function CronJobSidebarRuns({ jobId, onOpenRun }: { jobId: string; onOpenRun: (sessionId: string) => void }) {
  const { t } = useI18n()
  const c = t.cron
  const selectedSessionId = useStore($selectedStoredSessionId)
  const changeEventsAvailable = useStore($changeEventsAvailable)
  const cronChangeTick = useStore($cronChangeTick)
  const [runs, setRuns] = useState<null | SessionInfo[]>(null)
  const visible = usePaneVisible()

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

    // Hidden pane: skip the peek entirely — no initial load, no interval.
    // `visible` is in the dep array, so becoming visible re-runs this effect
    // and starts the load + timer fresh (same shape as the section clock).
    if (!visible) {
      return () => {
        cancelled = true
      }
    }

    void load()

    const intervalId = window.setInterval(
      () => {
        if (document.visibilityState === 'visible') {
          void load()
        }
      },
      changeEventsAvailable ? PEEK_BACKSTOP_INTERVAL_MS : PEEK_POLL_INTERVAL_MS
    )

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
    // cronChangeTick: a fired run reloads the peek immediately.
  }, [changeEventsAvailable, cronChangeTick, jobId, visible])

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
                'truncate rounded-md px-1.5 py-0.5 text-left text-[0.6875rem] tabular-nums focus-visible:bg-(--chrome-action-hover) focus-visible:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40',
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
