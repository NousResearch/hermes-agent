import { DEFAULT_REASONING_EFFORT } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { reasoningEffortClamp, reasoningEffortLabel } from '@/lib/reasoning-effort'
import { cn } from '@/lib/utils'
import { $defaultReasoningEffort } from '@/store/session'

import type { ChatBarState } from './types'

const PILL = cn(
  'h-(--composer-control-size) shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

/**
 * Composer reasoning selector: the active model's effort level as its own
 * pill next to the model pill, opening the same Thinking / Fast / Effort rows
 * the catalog offers per model — without having to find the model's row and
 * hover its submenu. Hidden when the catalog says the model has no reasoning
 * control, and while there is no live menu (gateway closed).
 *
 * Reads THIS surface's SessionView (primary or tile), like the model pill.
 */
export function ReasoningPill({ disabled, model }: { disabled: boolean; model: ChatBarState['model'] }) {
  const copy = useI18n().t.shell.modelOptions
  const view = useSessionView()
  const reasoningEffort = useStore(view.$reasoningEffort)
  const reasoningEffortWire = useStore(view.$reasoningEffortWire)
  const defaultEffort = useStore($defaultReasoningEffort)
  const [open, setOpen] = useState(false)

  if (!model.reasoningMenuContent || model.supportsReasoning === false) {
    return null
  }

  const effort = reasoningEffort || defaultEffort || DEFAULT_REASONING_EFFORT
  // A clamped pick (`ultra` → `max`) keeps the pill compact ("Ultra→Max") and
  // spells out the CLI's wording in the tooltip, so Ultra is never shown as a
  // distinct wire level the route does not have (#61634).
  const clamp = reasoningEffortClamp(effort, reasoningEffortWire)
  const label = reasoningEffortLabel(effort, reasoningEffortWire)

  const title = clamp
    ? `${copy.effort}: ${copy[clamp.effort]} (${copy.sendsOnRoute(copy[clamp.wire])})`
    : `${copy.effort}: ${label}`

  // Closing the menu ends its claim on the keyboard: Radix restores focus to
  // this pill (a toolbar button), so without the release the Enter that
  // committed a level also swallows whatever you type next.
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
          <Button
            aria-label={title}
            className={PILL}
            data-testid="reasoning-pill"
            disabled={disabled}
            type="button"
            variant="ghost"
          >
            <span>{label}</span>
            <ChevronDown className="size-2.5 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="w-52 p-0" side="top" sideOffset={8}>
        <ModelMenuCloseContext.Provider value={() => setMenuOpen(false)}>
          {model.reasoningMenuContent}
        </ModelMenuCloseContext.Provider>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
