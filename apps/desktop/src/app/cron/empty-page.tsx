import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/ui/empty-state'
import { useI18n } from '@/i18n'
import { usePortalShell } from '@/store/shell-variant'
import type { AutomationBlueprint } from '@/types/cron'

import { usePageFrameCopy } from '../overlays/page-frame-copy'
import { PanelHeader } from '../overlays/panel'
import { SettingsCard, SettingsGroup, SettingsLinkRow, SettingsOverview } from '../settings/settings-card'

interface CronEmptyPageProps {
  blueprints: readonly AutomationBlueprint[]
  onCreate: () => void
  onStartFrom: (blueprintKey: string) => void
}

/** The redesign's Schedules page before the first schedule: the empty card with its one action,
 *  then the blueprints as a list to start from. Nothing to search yet, so no search field. */
export function CronEmptyPage({ blueprints, onCreate, onStartFrom }: CronEmptyPageProps) {
  const { t } = useI18n()
  const c = t.cron

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <SettingsOverview>
        <EmptyState
          action={<Button onClick={onCreate}>{c.newCron}</Button>}
          description={c.emptyDescNew}
          title={c.emptyTitleNew}
        />
        {blueprints.length > 0 && (
          <SettingsGroup description={c.blueprints.subtitle} title={c.blueprints.tab}>
            <SettingsCard>
              {blueprints.map(item => (
                <SettingsLinkRow
                  description={item.description}
                  key={item.key}
                  meta={item.plugin || undefined}
                  onClick={() => onStartFrom(item.key)}
                  title={item.title}
                />
              ))}
            </SettingsCard>
          </SettingsGroup>
        )}
      </SettingsOverview>
    </div>
  )
}

/** The page heading. The redesign describes the page under its H1 and keeps the count as quiet
 *  meta, shown only once there is something to count (never "0 schedules" under the title). */
export function CronPageHeader({ totalCount }: { totalCount: number }) {
  const { t } = useI18n()
  const redesign = usePortalShell()
  const frame = usePageFrameCopy()
  const c = t.cron

  return redesign ? (
    <PanelHeader
      meta={totalCount > 0 ? c.count(totalCount) : undefined}
      subtitle={frame.schedulesDescription}
      title={c.title}
    />
  ) : (
    <PanelHeader subtitle={c.count(totalCount)} title={c.title} />
  )
}

/** The schedule list filtered to nothing: the shared empty block in the redesign. */
export function CronListEmpty({ searching }: { searching: boolean }) {
  const { t } = useI18n()
  const c = t.cron

  return usePortalShell() ? (
    <EmptyState description={c.emptyDescSearch} title={c.emptyTitleSearch} />
  ) : (
    <p className="px-2 py-4 text-center text-xs text-muted-foreground">
      {searching ? c.emptyTitleSearch : c.emptyTitleNew}
    </p>
  )
}
