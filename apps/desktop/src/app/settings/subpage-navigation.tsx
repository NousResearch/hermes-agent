import type { ReactNode } from 'react'

import { useI18n } from '@/i18n'

import { OverlayBreadcrumbHeader } from '../overlays/overlay-breadcrumb-header'
import type { OverlayNavGroup, OverlayNavLink } from '../overlays/overlay-split-layout'

export function SettingsSubpageHeader({
  child,
  group,
  trailing
}: {
  child?: OverlayNavLink
  group: OverlayNavGroup
  trailing?: ReactNode
}) {
  const { t } = useI18n()

  return (
    <OverlayBreadcrumbHeader
      child={child}
      group={group}
      rootLabel={t.commandCenter.settings}
      trailing={trailing}
    />
  )
}
