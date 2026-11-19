import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { type CronJob, getCronJobRuns, type SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { $changeEventsAvailable, $cronChangeTick } from '@/store/live-sync'

import { isCronRunRow, openCronRun, reconcileCronRunVerdicts } from '../cron/open-cron-run'

import { plainRunError } from './routine-error'
import { outputRunStatus, RUN_TONE, runChip, runDurationSeconds } from './routine-model'
import { useAgentLabel } from './routine-parts'
import { capitalized, formatDuration, formatMoment } from './routine-schedule'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'

// Runs come from the scheduler's tick: events re-read at once on backends that send them, so the poll
// is a slow backstop there; while a run is live it polls fast (the classic page's cadences).
const RUNS_POLL_MS = 8000
const RUNS_BACKSTOP_MS = 60_000
const LIVE_POLL_MS = 1500

/** The run history loads a batch at a time as it scrolls into view. The endpoint pages by `limit`
 *  alone (newest first, at most 100), so each batch re-reads the newest N, N growing by a batch. */
export const RUNS_BATCH = 20
export const RUNS_CAP = 100

interface RoutineRunsProps {
  job: CronJob
  live: boolean
  source: RoutineSource
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
}

/** What changes when the routine makes a new attempt or one settles. */
function runAttemptKey(job: CronJob): string {
  const execution = job.latest_execution

  return [execution?.id, execution?.status, job.last_run_at, job.last_status].map(part => part ?? '').join('|')
}

/** Run history (rev 1383:21029): When · Duration · Outcome, newest first, each run opening as a chat
 *  (`openCronRun`); its View / View error shows on hover, on keyboard focus, and always on touch. */
export function RoutineRuns({ job, live, onOpenSession, source }: RoutineRunsProps) {
  const copy = useRoutinesCopy()
  const changeEvents = useStore($changeEventsAvailable)
  const cronChangeTick = useStore($cronChangeTick)
  const scope = source.scopeFor(job)
  const [limit, setLimit] = useState(RUNS_BATCH)

  const runs = useQuery({
    // The routine row is the authority that an attempt happened: a run the scheduler refuses up front
    // records a failed attempt without a cron.changed event, so a new attempt is a new read. A bigger
    // batch keeps the shorter list on screen while it loads; another routine's never stands in.
    queryKey: ['routine-runs', source.connectionId ?? '', job.profile ?? '', job.id, runAttemptKey(job), limit],
    placeholderData: (previous, previousQuery) => (previousQuery?.queryKey[3] === job.id ? previous : undefined),
    queryFn: async () => {
      const result = await getCronJobRuns(job.id, limit, scope)
      // A fresh read re-evaluates every run already opened (#88443).
      reconcileCronRunVerdicts(result)

      return result
    },
    refetchInterval: live ? LIVE_POLL_MS : changeEvents ? RUNS_BACKSTOP_MS : RUNS_POLL_MS
  })

  const { refetch } = runs

  // A fired run moves jobs.json bookkeeping (cron.changed): re-read now.
  useEffect(() => {
    if (cronChangeTick > 0) {
      void refetch()
    }
  }, [cronChangeTick, refetch])

  const list = runs.data ?? []
  const starting = live && !list.some(run => runChip(run, job) === 'running')
  // A full batch may have more behind it; a short one is the whole history.
  const more = list.length >= limit && limit < RUNS_CAP
  const loadingMore = runs.isFetching && runs.isPlaceholderData

  return (
    <section aria-labelledby="routine-runs-title" className="routines-runs">
      <h2 className="routines-section-title" id="routine-runs-title">
        {copy.runs}
      </h2>
      {runs.isPending ? (
        <RunsSkeleton />
      ) : list.length === 0 && !starting ? (
        <p className="routines-runs-empty">{copy.noRuns}</p>
      ) : (
        <table className="routines-runs-table">
          <thead>
            <tr>
              <th scope="col">{copy.runColumns.when}</th>
              <th scope="col">{copy.runColumns.duration}</th>
              <th scope="col">{copy.runColumns.outcome}</th>
              <td />
            </tr>
          </thead>
          <tbody>
            {starting && <StartingRow />}
            {list.map(run => (
              <RunRow job={job} key={run.id} onOpenSession={onOpenSession} run={run} />
            ))}
          </tbody>
        </table>
      )}
      {more && <MoreRuns loading={loadingMore} onReach={() => setLimit(current => Math.min(current + RUNS_BATCH, RUNS_CAP))} />}
      {!more && list.length >= RUNS_CAP && <p className="routines-runs-note">{copy.newestRuns(RUNS_CAP)}</p>}
    </section>
  )
}

/** The end of the loaded history: reaching it (scrolled into view) loads the next batch. */
function MoreRuns({ loading, onReach }: { loading: boolean; onReach: () => void }) {
  const copy = useRoutinesCopy()
  const ref = useRef<HTMLParagraphElement>(null)
  const reach = useRef(onReach)
  reach.current = onReach

  useEffect(() => {
    const element = ref.current

    if (!element || loading) {
      return
    }

    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        reach.current()
      }
    })

    observer.observe(element)

    return () => observer.disconnect()
  }, [loading])

  return (
    <p aria-live="polite" className="routines-runs-note" data-runs-more="" ref={ref}>
      {loading ? copy.loadingMoreRuns : ' '}
    </p>
  )
}

function StartingRow() {
  const copy = useRoutinesCopy()

  return (
    <tr>
      <td>{copy.startingRun}</td>
      <td />
      <td>
        <span className="routines-chip" data-tone={RUN_TONE.running}>
          {copy.runChips.running}
        </span>
      </td>
      <td />
    </tr>
  )
}

interface RunRowProps {
  job: CronJob
  run: SessionInfo
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
}

function RunRow({ job, onOpenSession, run }: RunRowProps) {
  const { locale } = useI18n()
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()
  const [showError, setShowError] = useState(false)
  const chip = runChip(run, job)
  const seconds = runDurationSeconds(run)
  const when = capitalized(formatMoment((run.started_at || run.last_active) * 1000, locale), locale)
  const opens = isCronRunRow(run) && Boolean(onOpenSession)
  const failed = chip === 'failed' || chip === 'delivery' || chip === 'missed'
  // A run with no chat behind it says what went wrong in place.
  const reason = failed && !opens ? plainRunError(outputRunStatus(run)?.reason ?? run.preview ?? '', copy, agentLabel(job.profile ?? '')) : ''

  const action = opens ? (
    <Button onClick={() => onOpenSession && openCronRun(run, onOpenSession)} size="sm" variant="secondary">
      {failed ? copy.actions.viewError : copy.actions.view}
    </Button>
  ) : reason ? (
    <Button aria-expanded={showError} onClick={() => setShowError(!showError)} size="sm" variant="secondary">
      {copy.actions.viewError}
    </Button>
  ) : null

  return (
    <>
      <tr className="routines-runs-row">
        <td>{when}</td>
        <td className="routines-runs-muted">{seconds === null ? '—' : formatDuration(seconds, locale)}</td>
        <td>
          {chip ? (
            <span className="routines-chip" data-tone={RUN_TONE[chip]}>
              {copy.runChips[chip]}
            </span>
          ) : (
            <span className="routines-runs-muted">—</span>
          )}
        </td>
        <td className="routines-runs-action">{action}</td>
      </tr>
      {showError && reason && (
        <tr className="routines-runs-reason">
          <td colSpan={4}>{reason}</td>
        </tr>
      )}
    </>
  )
}

/** The loaded table's geometry while the first read is out. */
function RunsSkeleton() {
  return (
    <div aria-hidden className="routines-runs-skeleton">
      {Array.from({ length: 3 }, (_, index) => (
        <div className="routines-runs-skeleton-row" key={index}>
          <Skeleton className="h-3 w-32 max-w-full" />
          <Skeleton className="h-3 w-12 max-w-full" />
          <Skeleton className="h-3 w-16 max-w-full" />
        </div>
      ))}
    </div>
  )
}
