import { useI18n } from '@/i18n'

import { OverlayBreadcrumbHeader } from '../overlays/overlay-breadcrumb-header'
import type { OverlayNavGroup, OverlayNavLink } from '../overlays/overlay-split-layout'

export function SettingsSubpageHeader({ group, child }: { group: OverlayNavGroup; child?: OverlayNavLink }) {
  const { t } = useI18n()

  // A third rail level (a plugin's sub-page under Settings ▸ Plugins).
  const grandchild = child?.children?.find(link => link.active)

  return (
    <OverlayBreadcrumbHeader child={child} grandchild={grandchild} group={group} rootLabel={t.commandCenter.settings} />
  )
}
