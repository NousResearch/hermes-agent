import { type ReactNode, useEffect, useState } from 'react'
import { useLocation } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

/** Popover mounts the existing status surface only while open, including its polling. */
export function PortalStatusMenu({ children }: { children: (close: () => void) => ReactNode }) {
  const { t } = useI18n()
  const { pathname, search } = useLocation()
  const [open, setOpen] = useState(false)
  useEffect(() => setOpen(false), [pathname, search])
  const label = t.commandCenter.sections.system

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
        className="w-[300px] max-w-[calc(100vw-1rem)] max-h-[calc(100dvh-2rem)] overflow-y-auto p-0"
        side="right"
      >
        {children(() => setOpen(false))}
      </PopoverContent>
    </Popover>
  )
}
