import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { formatModelStatusLabel } from '@/lib/model-status-label'
import { cn } from '@/lib/utils'
import {
  $activeSessionId,
  $currentFastMode,
  $currentModel,
  $currentModelSource,
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
  const modelSource = useStore($currentModelSource)
  const activeSessionId = useStore($activeSessionId)
  const [open, setOpen] = useState(false)

  // The composer pick is sticky: a manual selection is pinned and every NEW
  // chat uses it instead of the Settings → Model default — silently, which has
  // cost users real money on a forgotten paid-model pick (#62055). Surface the
  // pin whenever a draft (no live session) is running on a manual override. A
  // live session's footer reflects that session's model, so no badge there.
  const pinnedOverride = !activeSessionId && modelSource === 'manual' && Boolean(currentModel.trim())

  // The model resolves a beat after the gateway/session comes up. Rather than
  // flash a literal "No model", show a quiet loader (inherits the pill text
  // color at half opacity) until a model lands.
  const label = compact ? (
    <ChevronDown className="size-3.5 shrink-0 opacity-70" />
  ) : (
    <>
      {currentModel.trim() ? (
        <span className="truncate">{formatModelStatusLabel(currentModel, { fastMode, reasoningEffort })}</span>
      ) : (
        <GlyphSpinner className="opacity-50" spinner="braille" />
      )}
      {pinnedOverride && (
        <span
          aria-label={copy.modelPinned}
          className="size-1 shrink-0 rounded-full bg-(--ui-accent)"
          data-testid="model-pinned-dot"
          role="img"
        />
      )}
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

  const baseTitle = currentProvider
    ? copy.modelTitle(currentProvider, currentModel || copy.modelNone)
    : copy.switchModel

  const title = pinnedOverride ? `${baseTitle} — ${copy.modelPinned}` : baseTitle

  if (!model.modelMenuContent) {
    return (
      <Tip label={pinnedOverride ? `${copy.openModelPicker} — ${copy.modelPinned}` : copy.openModelPicker} side="top">
        <Button
          aria-label={copy.openModelPicker}
          className={pillClass}
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
          <Button aria-label={title} className={pillClass} disabled={disabled} type="button" variant="ghost">
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
