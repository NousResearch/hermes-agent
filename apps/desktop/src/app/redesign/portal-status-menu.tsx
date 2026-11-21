import { type ReactNode, useEffect, useState } from 'react'
import { useLocation } from 'react-router'

import { useTitlebarHeight } from '@/app/shell/use-touch-titlebar'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

// The floating surfaces' own gutter (components/ui popover and menus).
const FLOATING_GUTTER = 8

/** Rail menus and popovers stay under the titlebar band and scroll inside the height left below it,
 *  so a short window (a landscape phone) never pushes them over the band or off the floor. */
export function useRailMenuCollisionPadding() {
  const top = useTitlebarHeight()

  return { top, right: FLOATING_GUTTER, bottom: FLOATING_GUTTER, left: FLOATING_GUTTER }
}

/** Popover mounts the existing status surface only while open, including its polling. */
export function PortalStatusMenu({ children }: { children: (close: () => void) => ReactNode }) {
  const { t } = useI18n()
  const { pathname, search } = useLocation()
  const [open, setOpen] = useState(false)
  useEffect(() => setOpen(false), [pathname, search])
  const label = t.commandCenter.sections.system
  const collisionPadding = useRailMenuCollisionPadding()

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <Tip label={label} placement="left-rail">
        <PopoverTrigger asChild>
          <Button aria-label={label} className="portal-chrome-button" size="icon" variant="ghost">
            <Codicon name="pulse" />
          </Button>
        </PopoverTrigger>
      </Tip>
      <PopoverContent
        align="end"
        aria-label={label}
        className="portal-rail-popover w-[300px] max-w-[calc(100vw-1rem)] max-h-[calc(100dvh-2rem)] overflow-y-auto p-0"
        collisionPadding={collisionPadding}
        side="right"
        sideOffset={0}
        // A rail popover drops beside its button like the rail's menus: no arrow.
        variant="menu"
      >
        {children(() => setOpen(false))}
      </PopoverContent>
    </Popover>
  )
}
