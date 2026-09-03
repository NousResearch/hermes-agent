import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useTourMarker } from '@/app/chat/tour-marker'
import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { modelStatusLabelParts } from '@/lib/model-status-label'
import { cn } from '@/lib/utils'
import { $currentModelSource, $defaultReasoningEffort, setModelPickerOpen } from '@/store/session'

import { onComposerModelMenuRequest } from './focus'
import { useComposerScope } from './scope'
import type { ChatBarState } from './types'

// `shrink` (not `shrink-0`) with a truncating label: the pill is the one
// control in the row that can give width back continuously, so it absorbs the
// squeeze between collapse stages instead of pushing Send past the edge.
const PILL = cn(
  'h-(--composer-control-size) min-w-0 max-w-40 shrink gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

/**
 * Composer model selector — the relocated status-bar pill. Reuses the live
 * `model.options` dropdown (`modelMenuContent`) verbatim; falls back to the
 * full picker when the gateway is closed and no live menu exists.
 *
 * Display follows THIS surface's SessionView (primary or tile) — never the
 * primary-only globals — so side-by-side panes each show their own model.
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
  // Two return branches below, one handle: only ever one of them mounts.
  const tourMarker = useTourMarker('model-pill')
  const view = useSessionView()
  // Prefer the chat-bar snapshot (already view-scoped by ChatView); fall back
  // to the live SessionView atoms so a mid-flight session.info still paints.
  const viewModel = useStore(view.$model)
  const viewProvider = useStore(view.$provider)
  const currentModel = model.model || viewModel
  const currentProvider = model.provider || viewProvider
  const fastMode = useStore(view.$fast)
  const reasoningEffort = useStore(view.$reasoningEffort)
  const modelSource = useStore($currentModelSource)
  const defaultEffort = useStore($defaultReasoningEffort)
  const runtimeId = useStore(view.$runtimeId)
  const [open, setOpen] = useState(false)
  const scope = useComposerScope()
  const hasLiveMenu = Boolean(model.modelMenuContent)

  // The `composer.modelPicker` hotkey, routed to exactly one surface (the pane
  // under the pointer, else the active composer — see requestModelMenuToggle).
  // Toggles the live dropdown; with no live menu (gateway closed) it opens the
  // full picker dialog, same as clicking the pill.
  useEffect(
    () =>
      onComposerModelMenuRequest(target => {
        if (target !== scope.target || disabled) {
          return
        }

        if (hasLiveMenu) {
          setOpen(prev => !prev)
        } else {
          setModelPickerOpen(true)
        }
      }),
    [scope.target, disabled, hasLiveMenu]
  )

  // The composer pick is sticky: a manual selection is pinned and every NEW
  // chat uses it instead of the Settings → Model default — silently, which has
  // cost users real money on a forgotten paid-model pick (#62055). Surface the
  // pin whenever a draft (no live session) is running on a manual override. A
  // live session's footer reflects that session's model, so no badge there.
  // Tiles always have a runtime — pin badge is primary-draft only.
  const pinnedOverride =
    view.kind === 'primary' && !runtimeId && modelSource === 'manual' && Boolean(currentModel.trim())

  // The model resolves a beat after the gateway/session comes up. Rather than
  // flash a literal "No model", show a quiet loader (inherits the pill text
  // color at half opacity) until a model lands.
  const modelParts = modelStatusLabelParts(currentModel, {
    defaultEffort,
    fastMode,
    provider: currentProvider,
    reasoningEffort
  })
  const label = compact ? (
    <ChevronDown className="size-3.5 shrink-0 opacity-70" />
  ) : (
    <>
      {currentModel.trim() ? (
        <span className="flex min-w-0 items-baseline gap-1.5 truncate">
          <span className="truncate">{modelParts.name}</span>
          {modelParts.provider && (
            <span className="truncate text-(--ui-text-quaternary)">{modelParts.provider}</span>
          )}
          {modelParts.meta && <span className="shrink-0 text-(--ui-text-quaternary)">{modelParts.meta}</span>}
        </span>
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
          data-tour={tourMarker}
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

  // Closing the menu ends its claim on the keyboard: Radix restores focus to
  // this pill (a toolbar button), so without the release the Enter that
  // committed a model also swallows whatever you type next.
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
            className={pillClass}
            data-tour={tourMarker}
            disabled={disabled}
            type="button"
            variant="ghost"
          >
            {label}
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="w-64 p-0" side="top" sideOffset={8}>
        <ModelMenuCloseContext.Provider value={() => setMenuOpen(false)}>
          {model.modelMenuContent}
        </ModelMenuCloseContext.Provider>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
