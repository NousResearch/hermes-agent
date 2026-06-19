import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { formatModelStatusLabel } from '@/lib/model-status-label'
import { cn } from '@/lib/utils'
import {
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  setModelPickerOpen
} from '@/store/session'

import type { ChatBarState } from './types'

const PILL = cn(
  'h-(--composer-control-size) max-w-40 shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

/**
 * Composer model selector — the relocated status-bar pill. Reuses the live
 * `model.options` dropdown (`modelMenuContent`) verbatim; falls back to the
 * full picker when the gateway is closed and no live menu exists.
 */
export function ModelPill({ disabled, model }: { disabled: boolean; model: ChatBarState['model'] }) {
  const copy = useI18n().t.shell.statusbar
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const fastMode = useStore($currentFastMode)
  const reasoningEffort = useStore($currentReasoningEffort)
  const [open, setOpen] = useState(false)

  // The model resolves a beat after the gateway/session comes up. Rather than
  // flash a literal "No model", show a quiet loader (inherits the pill text
  // color at half opacity) until a model lands.
  const label = (
    <>
      {currentModel.trim() ? (
        <span className="truncate">{formatModelStatusLabel(currentModel, { fastMode, reasoningEffort })}</span>
      ) : (
        <GlyphSpinner className="opacity-50" spinner="braille" />
      )}
      <ChevronDown className="size-2.5 shrink-0 opacity-50" />
    </>
  )

  const title =
    currentModel.trim() || currentProvider
      ? copy.modelTitle(currentProvider || copy.modelNone, currentModel || copy.modelNone)
      : copy.switchModel

  if (!model.modelMenuContent) {
    return (
      <Button
        aria-label={copy.openModelPicker}
        className={PILL}
        disabled={disabled}
        onClick={() => setModelPickerOpen(true)}
        title={title}
        type="button"
        variant="ghost"
      >
        {label}
      </Button>
    )
  }

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={copy.openModelPicker}
          className={PILL}
          disabled={disabled}
          title={title}
          type="button"
          variant="ghost"
        >
          {label}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64 p-0" side="top" sideOffset={8}>
        <ModelMenuCloseContext.Provider value={() => setOpen(false)}>
          {model.modelMenuContent}
        </ModelMenuCloseContext.Provider>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
