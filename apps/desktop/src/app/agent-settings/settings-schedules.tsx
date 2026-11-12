import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'

import { jobState } from '../cron/job-state'
import { ListRow, Pill } from '../settings/primitives'
import { SettingsAddRow, SettingsCard, SettingsEmptyRow, SettingsGroup } from '../settings/settings-card'
import { useSettingsPagesCopy } from '../settings/settings-pages-copy'

import { agentSettingsKey, type AgentSettingsOwner, getAgentRoutines } from './data'
import { RoutineDialog } from './routine-dialog'

const STATE_PILL: Record<string, 'destructive' | 'muted' | 'primary' | 'warn'> = {
  enabled: 'primary',
  scheduled: 'primary',
  running: 'primary',
  paused: 'warn',
  error: 'destructive'
}

/** Agent › Settings › Schedules (Figma 895:18509): one list card of this agent's schedules — state and
 *  Edit per row — ending in "+ New schedule". */
export function AgentSchedulesSettings({ owner }: { owner: AgentSettingsOwner }) {
  const { t } = useI18n()
  const pages = useSettingsPagesCopy()
  const [editor, setEditor] = useState<CronJob | 'new' | null>(null)
  const jobs = useQuery({ queryKey: [...agentSettingsKey(owner), 'routines'], queryFn: () => getAgentRoutines(owner) })

  if (jobs.isPending) {
    return <PageLoader />
  }

  if (jobs.isError) {
    return (
      <ErrorState description={jobs.error.message} title={t.cron.failedLoad}>
        <Button onClick={() => void jobs.refetch()} variant="secondary">
          {t.common.retry}
        </Button>
      </ErrorState>
    )
  }

  return (
    <>
      <SettingsGroup
        description={jobs.data.length ? t.cron.count(jobs.data.length) : pages.schedules.scheduledDesc}
        title={pages.schedules.scheduled}
      >
        <SettingsCard>
          {jobs.data.length === 0 && <SettingsEmptyRow title={t.cron.emptyTitleNew} />}
          {jobs.data.map(job => {
            const state = jobState(job)

            return (
              <ListRow
                action={
                  <>
                    <Pill tone={STATE_PILL[state] ?? 'muted'}>{t.cron.states[state] ?? state}</Pill>
                    <Button
                      aria-label={`${t.cron.edit}: ${job.name || job.id}`}
                      onClick={() => setEditor(job)}
                      size="sm"
                      variant="outline"
                    >
                      {t.skills.edit}
                    </Button>
                  </>
                }
                description={job.schedule_display || job.schedule?.display || job.schedule?.expr}
                key={job.id}
                title={job.name || job.id}
              />
            )
          })}
          <SettingsAddRow onClick={() => setEditor('new')}>{t.cron.createTitle}</SettingsAddRow>
        </SettingsCard>
      </SettingsGroup>
      {editor && (
        <RoutineDialog
          job={editor === 'new' ? undefined : editor}
          onClose={() => setEditor(null)}
          onSaved={() => void jobs.refetch()}
          owner={owner}
        />
      )}
    </>
  )
}
