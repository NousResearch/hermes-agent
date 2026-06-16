import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
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

  const label = (
    <>
      <span className="truncate">{formatModelStatusLabel(currentModel, { fastMode, reasoningEffort })}</span>
      <ChevronDown className="size-2.5 shrink-0 opacity-50" />
    </>
  )
  const title = currentProvider ? copy.modelTitle(currentProvider, currentModel || copy.modelNone) : copy.switchModel

  if (!model.modelMenuContent) {
    return (
      <Button
        aria-label={copy.openModelPicker}
        className={PILL}
        disabled={disabled}
        onClick={() => setModelPickerOpen(true)}
        title={copy.openModelPicker}
        type="button"
        variant="ghost"
      >
        {label}
      </Button>
    )
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button aria-label={title} className={PILL} disabled={disabled} title={title} type="button" variant="ghost">
          {label}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64 p-0" side="top" sideOffset={8}>
        {model.modelMenuContent}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
