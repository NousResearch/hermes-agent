import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { ChevronRight } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { PAGE_INSET_X } from '../layout-constants'

import type { OverlayNavGroup, OverlayNavLink } from './overlay-split-layout'

interface OverlayBreadcrumbHeaderProps {
  child?: OverlayNavLink
  group: OverlayNavGroup
  rootLabel: string
  trailing?: ReactNode
}

export function OverlayBreadcrumbHeader({
  child,
  group,
  rootLabel,
  trailing
}: OverlayBreadcrumbHeaderProps) {
  return (
    <div className={cn('mb-3 flex shrink-0 items-start justify-between gap-3', PAGE_INSET_X)}>
      <nav
        aria-label={group.label}
        className="flex min-w-0 items-center gap-1.5 text-xs text-(--ui-text-tertiary)"
      >
        <span className="shrink-0">{rootLabel}</span>
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
      {trailing && <div className="flex shrink-0 items-center">{trailing}</div>}
    </div>
  )
}
