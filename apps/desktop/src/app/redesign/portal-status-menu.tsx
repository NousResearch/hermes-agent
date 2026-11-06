import { type ReactNode, useEffect, useState } from 'react'
import { useLocation } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

/** Popover mounts the existing status surface only while open, including its polling. */
export function PortalStatusMenu({ children }: { children: ReactNode }) {
  const { t } = useI18n()
  const { pathname, search } = useLocation()
  const [open, setOpen] = useState(false)
  useEffect(() => setOpen(false), [pathname, search])
  const label = t.commandCenter.sections.system

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <Tip label={label}>
        <PopoverTrigger asChild>
          <Button aria-label={label} size="icon" variant="ghost">
            <Codicon name="pulse" />
          </Button>
        </PopoverTrigger>
      </Tip>
      <PopoverContent align="end" aria-label={label} className="w-80 max-w-[calc(100vw-1rem)]" side="right">
        <h2 className="px-2 py-1 text-sm font-semibold">{label}</h2>
        {children}
      </PopoverContent>
    </Popover>
  )
}
