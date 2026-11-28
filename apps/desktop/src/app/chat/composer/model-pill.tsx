import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useTourMarker } from '@/app/chat/tour-marker'
import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { isElementInHiddenPane } from '@/components/pane-shell/pane-visibility'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { formatModelPillLabel, providerDisplayName } from '@/lib/model-status-label'
import { useShortViewport } from '@/lib/short-viewport'
import { $touchPointer } from '@/lib/touch-interaction'
import { cn } from '@/lib/utils'
import { $currentModelSource, setModelPickerOpen } from '@/store/session'

import { useComposerModelPillLabel } from './contrib'
import { type ComposerTarget, onComposerModelMenuRequest } from './focus'
import { RICH_INPUT_SLOT } from './rich-editor'
import { useComposerScope } from './scope'
import type { ChatBarState } from './types'

/** Which composers currently have their model menu open, by composer scope target.
 *  A single shared boolean let one pane speak for every pane: opening a menu there kept a *different* pane's
 *  short-height collapse from minimizing, and unmounting any pill cleared the state for a menu that was still open.
 *  The owner is the composer target the hotkey and the focus bus already route by. */
export const $composerModelMenuOpen = atom<ReadonlySet<ComposerTarget>>(new Set())

/** Claim or release ``owner``'s menu. Rewrites the set so subscribers re-render on a real change only. */
export function setComposerModelMenuOpen(owner: ComposerTarget, open: boolean): void {
  const owners = $composerModelMenuOpen.get()

  if (owners.has(owner) === open) {
    return
  }

  const next = new Set(owners)

  if (open) {
    next.add(owner)
  } else {
    next.delete(owner)
  }

  $composerModelMenuOpen.set(next)
}

/** Whether THIS composer's model menu is open. */
export function useComposerModelMenuOpen(owner: ComposerTarget): boolean {
  return useStore($composerModelMenuOpen).has(owner)
}

// `shrink` (not `shrink-0`) with a truncating label: the pill is the one
// control in the row that can give width back continuously, so it absorbs the
// squeeze between collapse stages instead of pushing Send past the edge.
// No `max-w-*` cap: the pill sizes to its label, so a long model name only
// truncates when the row is genuinely out of room (#49340) — not at an
// arbitrary 160px.
const UNKNOWN_TIER = atom('')

const PILL = cn(
  'h-(--composer-control-size) min-w-0 shrink gap-1 rounded-md px-2 text-xs font-normal',
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
  const touchPointer = useStore($touchPointer)
  const short = useShortViewport()
  // The phone footer may yield its model label to quota values. Keep the
  // current choice readable normally; keyboard-height collapse keeps its chevron.
  const iconOnly = compact && (!touchPointer || short)
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
  const serviceTier = useStore(view.$serviceTier ?? UNKNOWN_TIER)
  const reasoningEffort = useStore(view.$reasoningEffort)
  const modelSource = useStore($currentModelSource)
  const runtimeId = useStore(view.$runtimeId)
  const scope = useComposerScope()
  const [open, setOpen] = useState(false)
  // Mirrors `open` for the hotkey handler, so the toggle reads the live value without re-subscribing per render.
  const openRef = useRef(false)

  // Every transition goes through here — Radix's onOpenChange and the hotkey alike — so this composer's claim can
  // never disagree with the mounted menu. A hotkey that only set `open` left the claim unset, and the short-height
  // row then unmounted the menu it had just opened.
  const setMenuOpen = useCallback(
    (next: boolean) => {
      openRef.current = next
      setOpen(next)
      setComposerModelMenuOpen(scope.target, next)

      // Closing the menu ends its claim on the keyboard: Radix restores focus to this pill (a toolbar button), so
      // without the release the Enter that committed a model also swallows whatever you type next.
      if (!next) {
        releaseTypingFocus()
      }
    },
    [scope.target]
  )

  // An unmount releases its OWN claim and no other: clearing the whole registry here closed a state that a
  // neighbouring composer's still-open menu depended on.
  useEffect(() => () => setComposerModelMenuOpen(scope.target, false), [scope.target])
  const restoreSelection = useRef<(() => void) | null>(null)
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
          const editor = document.activeElement
          const selection = window.getSelection()

          if (
            editor instanceof HTMLElement &&
            editor.dataset.slot === RICH_INPUT_SLOT &&
            selection?.anchorNode &&
            selection.focusNode &&
            editor.contains(selection.anchorNode) &&
            editor.contains(selection.focusNode)
          ) {
            const { anchorNode, anchorOffset, focusNode, focusOffset } = selection

            restoreSelection.current = () => {
              if (
                !editor.isConnected ||
                isElementInHiddenPane(editor) ||
                !editor.contains(anchorNode) ||
                !editor.contains(focusNode)
              ) {
                return
              }

              editor.focus({ preventScroll: true })
              window.getSelection()?.setBaseAndExtent(anchorNode, anchorOffset, focusNode, focusOffset)
            }
          }

          setMenuOpen(!openRef.current)
        } else {
          setModelPickerOpen(true)
        }
      }),
    [disabled, hasLiveMenu, scope.target, setMenuOpen]
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
  //
  // A `composer.modelPill` provider may override the LABEL (compact reasoning
  // label, custom naming) — the pill keeps its chrome, pin dot, and menu; only
  // the text changes, and a provider that declines leaves the core label.
  const pillLabel = useComposerModelPillLabel({ compact, model: currentModel, reasoningEffort: reasoningEffort || '' })

  const label = iconOnly ? (
    <ChevronDown className="size-3.5 shrink-0 opacity-70" />
  ) : (
    <>
      {currentModel.trim() ? (
        <span className="truncate">
          {pillLabel ??
            formatModelPillLabel(currentModel, {
              fastMode,
              requireCapabilities: !runtimeId,
              serviceTier,
              supportsFast: model.supportsFast,
              supportsUltrafast: model.supportsUltrafast
            })}
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

  // Non-touch compact mode keeps its snug icon button. Touch uses the normal
  // truncating/shrinking pill, so Send and the other controls keep their room.
  const pillClass = iconOnly
    ? cn(
        'size-(--composer-control-size) shrink-0 justify-center gap-0 rounded-md p-0',
        'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      )
    : PILL

  const baseTitle = currentProvider
    ? copy.modelTitle(providerDisplayName(currentProvider), currentModel || copy.modelNone)
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
      <DropdownMenuContent
        align="end"
        className="w-72 p-0"
        onCloseAutoFocus={event => {
          if (restoreSelection.current) {
            event.preventDefault()
            restoreSelection.current()
            restoreSelection.current = null
          }
        }}
        onInteractOutside={() => {
          restoreSelection.current = null
        }}
        side="top"
        sideOffset={8}
      >
        <ModelMenuCloseContext.Provider value={() => setMenuOpen(false)}>
          {model.modelMenuContent}
        </ModelMenuCloseContext.Provider>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
