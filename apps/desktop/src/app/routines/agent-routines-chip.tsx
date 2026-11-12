import { useStore } from '@nanostores/react'
import { useNavigate } from 'react-router'

import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'
import { agentSidebarKey } from '@/store/agent-sidebar-preferences'
import { $cronJobs, getCronJobsSnapshotScope } from '@/store/cron'

import { CRON_ROUTE } from '../routes'

import { routineNeedsAttention, routineTitle } from './routine-model'
import { useRoutinesCopy } from './routines-copy'
import { $routineOverviewQuery, selectRoutine } from './routines-store'

export interface AgentRoutineSummary {
  count: number
  failing: CronJob[]
}

/** Each agent's routines from the shared snapshot, keyed like the chat sidebar's agents. A snapshot
 *  with no owning scope (wiped by a switch) says nothing about anyone's routines. */
export function agentRoutineSummaries(jobs: readonly CronJob[], scope: string): Map<string, AgentRoutineSummary> {
  const summaries = new Map<string, AgentRoutineSummary>()

  if (!scope) {
    return summaries
  }

  const [connection, listed] = scope.split('\u0000')

  for (const job of jobs) {
    const profile = listed === 'all' ? job.profile : listed

    if (!profile) {
      continue
    }

    const key = agentSidebarKey({ connectionId: connection || null, profile })
    const summary = summaries.get(key) ?? { count: 0, failing: [] }
    summary.count += 1

    if (routineNeedsAttention(job)) {
      summary.failing.push(job)
    }

    summaries.set(key, summary)
  }

  return summaries
}

export function useAgentRoutineSummaries(): Map<string, AgentRoutineSummary> {
  return agentRoutineSummaries(useStore($cronJobs), getCronJobsSnapshotScope())
}

interface AgentRoutinesChipProps {
  agentLabel: string
  summary?: AgentRoutineSummary
}

/** R5 (Figma 1100:8452): under an agent's chats, how many routines it has and a failure dot; it opens
 *  the Routines page on that agent's routines. Runs themselves never join Chats. Nothing for an agent
 *  with no routines. */
export function AgentRoutinesChip({ agentLabel, summary }: AgentRoutinesChipProps) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()
  const navigate = useNavigate()

  if (!summary?.count) {
    return null
  }

  const failing = summary.failing[0]
  const detail = failing ? copy.chipFailing(routineTitle(failing)) : copy.chipOpens

  return (
    <div className="portal-agent-routines">
      <Tip
        label={
          <span className="portal-agent-routines-tip">
            <span>{copy.chipTooltip(summary.count, agentLabel)}</span>
            <span>{detail}</span>
          </span>
        }
      >
        <button
          aria-description={detail}
          aria-label={`${t.cron.title} · ${copy.chipTooltip(summary.count, agentLabel)}`}
          className="portal-agent-routines-chip"
          data-attention={failing ? '' : undefined}
          onClick={() => {
            selectRoutine(null)
            $routineOverviewQuery.set(agentLabel)
            navigate(CRON_ROUTE)
          }}
          type="button"
        >
          <Codicon aria-hidden name="clock" size="0.75rem" />
          <span>{t.cron.title}</span>
          <span className="portal-agent-routines-count">{summary.count}</span>
          {failing && <span aria-hidden className="portal-agent-routines-dot" />}
          <Codicon aria-hidden className="portal-agent-routines-chevron" name="chevron-right" size="0.75rem" />
        </button>
      </Tip>
    </div>
  )
}
