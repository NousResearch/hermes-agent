import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { AlertTriangle, ChevronDown } from '@/lib/icons'
import { formatModelStatusLabel } from '@/lib/model-status-label'
import { cn } from '@/lib/utils'
import {
  $activeSessionId,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $profileDefaultModel,
  $profileDefaultProvider,
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
export function ModelPill({
  compact = false,
  disabled,
  model
}: {
  compact?: boolean
  disabled: boolean
  model: ChatBarState['model']
}) {
  const copy = useI18n().t.shell.statusbar
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const fastMode = useStore($currentFastMode)
  const reasoningEffort = useStore($currentReasoningEffort)
  const activeSessionId = useStore($activeSessionId)
  const profileDefaultModel = useStore($profileDefaultModel)
  const profileDefaultProvider = useStore($profileDefaultProvider)
  const [open, setOpen] = useState(false)

  const normalizedCurrentModel = currentModel.trim().toLowerCase()
  const normalizedDefaultModel = profileDefaultModel.trim().toLowerCase()
  const normalizedCurrentProvider = currentProvider.trim().toLowerCase()
  const normalizedDefaultProvider = profileDefaultProvider.trim().toLowerCase()

  const providerDiffers =
    !!normalizedCurrentProvider &&
    !!normalizedDefaultProvider &&
    normalizedCurrentProvider !== normalizedDefaultProvider

  const hasComposerOverride =
    !activeSessionId &&
    !!normalizedCurrentModel &&
    !!normalizedDefaultModel &&
    (normalizedCurrentModel !== normalizedDefaultModel || providerDiffers)

  // The model resolves a beat after the gateway/session comes up. Rather than
  // flash a literal "No model", show a quiet loader (inherits the pill text
  // color at half opacity) until a model lands.
  const overrideIcon = hasComposerOverride ? (
    <AlertTriangle aria-hidden="true" className="size-3 shrink-0 text-amber-500" data-slot="model-override-indicator" />
  ) : null

  const label = compact ? (
    (overrideIcon ?? <ChevronDown className="size-3.5 shrink-0 opacity-70" />)
  ) : (
    <>
      {currentModel.trim() ? (
        <span className="truncate">{formatModelStatusLabel(currentModel, { fastMode, reasoningEffort })}</span>
      ) : (
        <GlyphSpinner className="opacity-50" spinner="braille" />
      )}
      {overrideIcon}
      <ChevronDown className="size-2.5 shrink-0 opacity-50" />
    </>
  )

  // Compact (floating composer): a snug square holding just the chevron — no pill
  // padding, sized to match the other composer icon buttons.
  const pillClass = compact
    ? cn(
        'size-(--composer-control-size) shrink-0 justify-center gap-0 rounded-md p-0',
        'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      )
    : PILL

  const title = hasComposerOverride
    ? copy.modelOverrideTitle(
        currentProvider || copy.unknown,
        currentModel,
        profileDefaultProvider || copy.unknown,
        profileDefaultModel
      )
    : currentProvider
      ? copy.modelTitle(currentProvider, currentModel || copy.modelNone)
      : copy.switchModel

  if (!model.modelMenuContent) {
    return (
      <Tip label={hasComposerOverride ? title : copy.openModelPicker} side="top">
        <Button
          aria-label={hasComposerOverride ? title : copy.openModelPicker}
          className={pillClass}
          data-model-override={hasComposerOverride ? 'true' : undefined}
          disabled={disabled}
          onClick={() => setModelPickerOpen(true)}
          type="button"
          variant="ghost"
        >
          {label}
        </Button>
      </Tip>
    )
  }

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <Tip label={title} side="top">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={title}
            className={pillClass}
            data-model-override={hasComposerOverride ? 'true' : undefined}
            disabled={disabled}
            type="button"
            variant="ghost"
          >
            {label}
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="w-64 p-0" side="top" sideOffset={8}>
        <ModelMenuCloseContext.Provider value={() => setOpen(false)}>
          {model.modelMenuContent}
        </ModelMenuCloseContext.Provider>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
