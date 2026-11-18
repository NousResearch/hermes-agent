import { useI18n } from '@/i18n'

import { PanelHeader } from '../overlays/panel'

/** The classic Scheduled jobs heading: the title over the job count. */
export function CronPageHeader({ totalCount }: { totalCount: number }) {
  const { t } = useI18n()
  const c = t.cron

  return <PanelHeader subtitle={c.count(totalCount)} title={c.title} />
}

/** The job list filtered to nothing (classic's quiet line under the list). */
export function CronListEmpty({ searching }: { searching: boolean }) {
  const { t } = useI18n()
  const c = t.cron

  return (
    <p className="px-2 py-4 text-center text-xs text-muted-foreground">
      {searching ? c.emptyTitleSearch : c.emptyTitleNew}
    </p>
  )
}
