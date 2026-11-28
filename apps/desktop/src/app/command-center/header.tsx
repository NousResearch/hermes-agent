
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { usePortalShell } from '@/store/shell-variant'

import { OverlayBreadcrumbHeader } from '../overlays/overlay-breadcrumb-header'
import type { OverlayNavGroup } from '../overlays/overlay-split-layout'

const USAGE_PERIODS = [7, 30, 90] as const
export type UsagePeriod = (typeof USAGE_PERIODS)[number]

interface CommandCenterHeaderProps {
  group: OverlayNavGroup
  onPeriodChange: (period: UsagePeriod) => void
  onQueryChange: (query: string) => void
  period: UsagePeriod
  query: string
}

export function CommandCenterHeader({ group, onPeriodChange, onQueryChange, period, query }: CommandCenterHeaderProps) {
  const { t } = useI18n()
  const simplified = usePortalShell()

  return (
    <OverlayBreadcrumbHeader
      group={group}
      rootLabel={t.commandCenter.commandCenter}
      trailing={
        <>
          {group.id === 'sessions' && (
            <SearchField
              containerClassName="max-w-[40vw] [[data-page-surface]_&]:max-w-full"
              onChange={onQueryChange}
              placeholder={t.commandCenter.searchPlaceholder}
              value={query}
              variant={simplified ? 'box' : 'underline'}
            />
          )}
          {group.id === 'usage' && (
            <SegmentedControl
              onChange={id => onPeriodChange(Number(id) as UsagePeriod)}
              options={USAGE_PERIODS.map(value => ({ id: String(value), label: t.commandCenter.days(value) }))}
              value={String(period)}
            />
          )}
        </>
      }
    />
  )
}
