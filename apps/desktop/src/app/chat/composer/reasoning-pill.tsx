import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { DEFAULT_REASONING_EFFORT, reasoningEffortLabel } from '@/lib/reasoning-effort'
import { cn } from '@/lib/utils'
import { $defaultReasoningEffort } from '@/store/session'

import type { ChatBarState } from './types'

const PILL = cn(
  'h-(--composer-control-size) min-w-0 shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

/** Composer control dedicated to the active model's reasoning effort. */
export function ReasoningPill({ disabled, model }: { disabled: boolean; model: ChatBarState['model'] }) {
  const copy = useI18n().t.shell.modelOptions
  const [open, setOpen] = useState(false)
  const profileDefault = useStore($defaultReasoningEffort)
  const effort = model.reasoningEffort || profileDefault || DEFAULT_REASONING_EFFORT
  const label = reasoningEffortLabel(effort)
  const title = `${copy.effort}: ${label}`

  if (!model.reasoningMenuContent || model.supportsReasoning === false) {
    return null
  }

  const setMenuOpen = (next: boolean) => {
    setOpen(next)

    if (!next) {
      releaseTypingFocus()
    }
  }

  return (
    <DropdownMenu onOpenChange={setMenuOpen} open={open}>
      <Tip label={title} side="top">
        <DropdownMenuTrigger asChild>
          <Button aria-label={title} className={PILL} disabled={disabled} type="button" variant="ghost">
            <span>{label}</span>
            <ChevronDown className="size-2.5 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="start" className="w-48" side="top">
        <ModelMenuCloseContext.Provider value={() => setMenuOpen(false)}>
          {model.reasoningMenuContent}
        </ModelMenuCloseContext.Provider>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
