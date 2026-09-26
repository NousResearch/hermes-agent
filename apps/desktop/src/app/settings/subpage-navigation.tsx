import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { ChevronRight } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { PAGE_INSET_X } from '../layout-constants'
import type { OverlayNavGroup, OverlayNavLink } from '../overlays/overlay-split-layout'

export function SettingsSubpageHeader({ group, child }: { group: OverlayNavGroup; child?: OverlayNavLink }) {
  const { t } = useI18n()

  return (
    <nav
      aria-label={group.label}
      className={cn('mb-3 flex shrink-0 items-center gap-1.5 text-xs text-(--ui-text-tertiary)', PAGE_INSET_X)}
    >
      <span className="shrink-0">{t.commandCenter.settings}</span>
      <ChevronRight aria-hidden className="size-3 shrink-0" />
      {child ? (
        <>
          <Button onClick={group.onSelect} size="inline" variant="text">
            {group.label}
          </Button>
          <ChevronRight aria-hidden className="size-3 shrink-0" />
          <span aria-current="page" className="truncate text-foreground">
            {child.label}
          </span>
        </>
      ) : (
        <span aria-current="page" className="truncate text-foreground">
          {group.label}
        </span>
      )}
    </nav>
  )
}
