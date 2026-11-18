import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { type CronJob, getCronJobRuns, type SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { $changeEventsAvailable, $cronChangeTick } from '@/store/live-sync'

import { isCronRunRow, openCronRun, reconcileCronRunVerdicts } from '../cron/open-cron-run'
import { ListRow } from '../settings/primitives'
import { SettingsCard, SettingsCardSkeleton, SettingsEmptyRow, SettingsGroup } from '../settings/settings-card'

import { plainRunError } from './routine-error'
import { outputRunStatus, runChip, runDurationSeconds } from './routine-model'
import { useAgentLabel } from './routine-parts'
import { capitalized, formatDuration, formatMoment } from './routine-schedule'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'

// Runs come from the scheduler's tick: events re-read at once on backends that send them, so the poll
// is a slow backstop there; while a run is live it polls fast (the classic page's cadences).
const RUNS_POLL_MS = 8000
const RUNS_BACKSTOP_MS = 60_000
const LIVE_POLL_MS = 1500

interface RoutineRunsProps {
  job: CronJob
  live: boolean
  source: RoutineSource
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
}

/** R2's Runs: newest first, each run opens as a chat (the existing run → session door, `openCronRun`).
 *  Runs never join the Chats sidebar; this card is where they live. */
export function RoutineRuns({ job, live, onOpenSession, source }: RoutineRunsProps) {
  const { locale } = useI18n()
  const copy = useRoutinesCopy()
  const changeEvents = useStore($changeEventsAvailable)
  const cronChangeTick = useStore($cronChangeTick)
  const scope = source.scopeFor(job)

  const runs = useQuery({
    // The routine row is the authority that an attempt happened: a run the scheduler refuses up front
    // (its agent not running here) records a failed attempt and its output without a cron.changed
    // event, so a new attempt is a new read rather than a wait for the slow poll. This routine's list
    // stays on screen while it loads; another routine's never stands in for it.
    queryKey: ['routine-runs', source.connectionId ?? '', job.profile ?? '', job.id, runAttemptKey(job)],
    placeholderData: (previous, previousQuery) => (previousQuery?.queryKey[3] === job.id ? previous : undefined),
    queryFn: async () => {
      const result = await getCronJobRuns(job.id, 20, scope)
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

  return (
    <SettingsGroup description={copy.runsHint} title={copy.runs}>
      {runs.isPending ? (
        <SettingsCardSkeleton rows={3} />
      ) : (
        <SettingsCard>
          {starting && <ListRow description={copy.startingRun} title={copy.runChips.running} />}
          {list.length === 0 && !starting && <SettingsEmptyRow title={copy.noRuns} />}
          {list.map(run => (
            <RunRow job={job} key={run.id} locale={locale} onOpenSession={onOpenSession} run={run} />
          ))}
        </SettingsCard>
      )}
    </SettingsGroup>
  )
}

interface RunRowProps {
  job: CronJob
  locale: string
  run: SessionInfo
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
}

/** What changes when the routine makes a new attempt or one settles. */
function runAttemptKey(job: CronJob): string {
  const execution = job.latest_execution

  return [execution?.id, execution?.status, job.last_run_at, job.last_status].map(part => part ?? '').join('|')
}

function RunRow({ job, locale, onOpenSession, run }: RunRowProps) {
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()
  const plain = (raw: string) => plainRunError(raw, copy, agentLabel(job.profile ?? ''))
  const chip = runChip(run, job)
  const seconds = runDurationSeconds(run)
  const when = capitalized(formatMoment((run.started_at || run.last_active) * 1000, locale), locale)

  const output = outputRunStatus(run)

  const description =
    chip === 'running'
      ? copy.runningNow
      : output
        ? plain(output.reason)
        : seconds !== null && run.source === 'cron'
          ? copy.finishedIn(formatDuration(seconds, locale))
          : plain(run.title || run.preview || '')

  return (
    <ListRow
      action={
        (chip || (isCronRunRow(run) && onOpenSession)) && (
          <>
            {chip && (
              <span className="routines-chip" data-tone={chip === 'running' ? 'running' : chip === 'stopped' ? 'muted' : 'bad'}>
                {copy.runChips[chip]}
              </span>
            )}
            {isCronRunRow(run) && onOpenSession && (
              <Button onClick={() => openCronRun(run, onOpenSession)} variant="secondary">
                {copy.actions.view}
              </Button>
            )}
          </>
        )
      }
      description={description}
      title={when}
    />
  )
}
