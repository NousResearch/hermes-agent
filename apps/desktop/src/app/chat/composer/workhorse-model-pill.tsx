import { useState } from 'react'

import { ModelMenuCloseContext } from '@/app/shell/model-catalog-menu'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Users } from '@/lib/icons'

import { GHOST_ICON_BTN } from './control-classes'
import { useWorkhorseConfig } from './workhorse-config'
import { WorkhorseMenuContent } from './workhorse-menu-content'

export interface WorkhorseModelPillProps {
  disabled: boolean
}

/** One icon at every composer width; catalog and settings mount only on open. */
export function WorkhorseModelPill({ disabled }: WorkhorseModelPillProps) {
  const { t } = useI18n()
  const config = useWorkhorseConfig()
  const [open, setOpen] = useState(false)

  const setMenuOpen = (next: boolean) => {
    setOpen(next)

    if (!next) {
      releaseTypingFocus()
    }
  }

  return (
    <DropdownMenu onOpenChange={setMenuOpen} open={open}>
      <Tip label={t.shell.statusbar.workhorse.label} side="top">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={t.shell.statusbar.workhorse.label}
            className={GHOST_ICON_BTN}
            data-testid="workhorse-model-pill"
            disabled={disabled || !config.ready}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Users aria-hidden="true" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent
        align="end"
        className="w-72 p-0"
        data-testid="workhorse-model-menu"
        side="top"
        sideOffset={8}
      >
        <ModelMenuCloseContext.Provider value={() => setMenuOpen(false)}>
          <WorkhorseMenuContent {...config} />
        </ModelMenuCloseContext.Provider>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
