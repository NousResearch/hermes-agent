import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { ChevronRight } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { usePortalShell } from '@/store/shell-variant'

import { PAGE_INSET_X } from '../layout-constants'

import type { OverlayNavGroup, OverlayNavLink } from './overlay-split-layout'

interface OverlayBreadcrumbHeaderProps {
  child?: OverlayNavLink
  /** A child's own sub-page (third nav level). */
  grandchild?: OverlayNavLink
  group: OverlayNavGroup
  rootLabel: string
  trailing?: ReactNode
}

export function OverlayBreadcrumbHeader({
  child,
  grandchild,
  group,
  rootLabel,
  trailing
}: OverlayBreadcrumbHeaderProps) {
  const simplified = usePortalShell()

  return (
    <header className={cn('shrink-0', simplified ? 'mb-6' : 'mb-3', PAGE_INSET_X)} data-slot="page-heading">
      <div className="flex items-start justify-between gap-3">
        <nav aria-label={group.label} className={cn('flex min-w-0 items-center gap-1.5 text-xs text-(--ui-text-tertiary)', simplified && !child && 'hidden')}>
          <span className="shrink-0">{rootLabel}</span>
          <ChevronRight aria-hidden className="size-3 shrink-0" />
          {child ? (
            <>
              <Button onClick={group.onSelect} size="inline" variant="text">
                {group.label}
              </Button>
              <ChevronRight aria-hidden className="size-3 shrink-0" />
              {grandchild ? (
                <>
                  <Button className="min-w-0 truncate" onClick={child.onSelect} size="inline" variant="text">
                    {child.label}
                  </Button>
                  <ChevronRight aria-hidden className="size-3 shrink-0" />
                  <span aria-current="page" className="truncate text-foreground">
                    {grandchild.label}
                  </span>
                </>
              ) : (
                <span aria-current="page" className="truncate text-foreground">
                  {child.label}
                </span>
              )}
            </>
          ) : (
            <span aria-current="page" className="truncate text-foreground">
              {group.label}
            </span>
          )}
        </nav>
        {!simplified && trailing && <div className="flex shrink-0 items-center">{trailing}</div>}
      </div>
      {simplified && (
        <div className="mt-3 flex flex-wrap items-center justify-between gap-4">
          <h1 className="min-w-0 text-3xl leading-tight font-medium tracking-tight text-foreground">
            {grandchild?.label ?? child?.label ?? group.label}
          </h1>
          {trailing && <div className="flex min-w-0 flex-wrap items-center gap-2">{trailing}</div>}
        </div>
      )}
    </header>
  )
}
