import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'

import { ListRow } from '../settings/primitives'

import { agentSettingsKey, type AgentSettingsOwner, getAgentRoutines } from './data'
import { RoutineDialog } from './routine-dialog'

export function AgentSchedulesSettings({ owner }: { owner: AgentSettingsOwner }) {
  const { t } = useI18n()
  const [editor, setEditor] = useState<CronJob | 'new' | null>(null)
  const jobs = useQuery({ queryKey: [...agentSettingsKey(owner), 'routines'], queryFn: () => getAgentRoutines(owner) })

  return (
    <>
      <Button onClick={() => setEditor('new')}>{t.cron.createTitle}</Button>
      {jobs.isPending ? (
        <PageLoader />
      ) : jobs.isError ? (
        <ErrorState description={jobs.error.message} title={t.cron.failedLoad}>
          <Button onClick={() => void jobs.refetch()} variant="secondary">
            {t.common.retry}
          </Button>
        </ErrorState>
      ) : jobs.data.length ? (
        <div className="agent-settings-card">
          {jobs.data.map(job => (
            <ListRow
              action={
                <Button onClick={() => setEditor(job)} variant="secondary">
                  {t.cron.edit}
                </Button>
              }
              description={job.schedule_display || job.schedule?.display || job.schedule?.expr}
              key={job.id}
              title={job.name || job.id}
            />
          ))}
        </div>
      ) : (
        <p>{t.cron.emptyTitleNew}</p>
      )}
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
