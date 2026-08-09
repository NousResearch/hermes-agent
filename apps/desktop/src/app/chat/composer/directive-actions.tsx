/**
 * Hover actions for directive chips in a composer.
 *
 * A directive chip (`@url:`, `@session:`, …) reads as the thing it points at
 * and is coloured like one, but a composer is an editor — a click inside the
 * contenteditable only places the caret, so there's no way to *act* on the
 * reference. Instead, hovering a chip whose kind has an action floats a small
 * pill above it that runs it.
 *
 * The kind → action table (`DIRECTIVE_ACTIONS`) lives in `directive-text`, so
 * it is shared with the sent-message chip: one entry lights up both surfaces.
 */
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { DIRECTIVE_ACTIONS, type DirectiveAction } from '@/components/assistant-ui/directive-text'
import { composerFloatingPill } from '@/components/chat/composer-dock'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

/** Moving between the chip and the pill crosses a gap where neither is hovered.
 *  Short enough that it still reads as instant on the way out. */
const HIDE_DELAY_MS = 120

/** The actionable directive chip under `target` that also belongs to `editor`,
 *  if there is one. */
function actionableChipAt(target: EventTarget | null, editor: HTMLElement): HTMLElement | null {
  const chip = target instanceof Element ? target.closest<HTMLElement>('[data-ref-kind]') : null
  const kind = chip?.dataset.refKind

  return chip && kind && chip.dataset.refId && editor.contains(chip) && DIRECTIVE_ACTIONS[kind] ? chip : null
}

interface Anchor {
  action: DirectiveAction
  chip: HTMLElement
  left: number
  top: number
  value: string
}

function anchorFor(chip: HTMLElement): Anchor | null {
  const value = chip.dataset.refId
  const action = chip.dataset.refKind ? DIRECTIVE_ACTIONS[chip.dataset.refKind] : undefined

  if (!value || !action || !chip.isConnected) {
    return null
  }

  const rect = chip.getBoundingClientRect()

  return { action, chip, left: rect.left, top: rect.top, value }
}

/**
 * Renders the action pill for whichever actionable chip in `editorRef` is
 * hovered.
 *
 * Listeners bind to `document`, not the editor, so mount timing can't strand
 * them: the edit composer's contenteditable isn't reliably attached when this
 * effect first runs, and a document listener that reads the editor lazily works
 * regardless. Each instance filters to its own editor, so the docked and edit
 * composers never show two pills for one chip.
 */
export function ComposerDirectiveActions({ editorRef }: { editorRef: RefObject<HTMLElement | null> }) {
  const { t } = useI18n()
  const [anchor, setAnchor] = useState<Anchor | null>(null)
  const hideTimerRef = useRef<number | undefined>(undefined)

  const cancelHide = useCallback(() => {
    window.clearTimeout(hideTimerRef.current)
  }, [])

  const hideSoon = useCallback(() => {
    cancelHide()
    hideTimerRef.current = window.setTimeout(() => setAnchor(null), HIDE_DELAY_MS)
  }, [cancelHide])

  useEffect(() => {
    const onPointerOver = (event: PointerEvent) => {
      const editor = editorRef.current
      const chip = editor && actionableChipAt(event.target, editor)

      if (!chip) {
        return
      }

      cancelHide()
      setAnchor(current => (current?.chip === chip ? current : anchorFor(chip)))
    }

    const onPointerOut = (event: PointerEvent) => {
      const editor = editorRef.current
      const chip = editor && actionableChipAt(event.target, editor)

      // A move within the same chip (its icon → its label) is not a leave.
      if (chip && editor && chip === actionableChipAt(event.relatedTarget, editor)) {
        return
      }

      hideSoon()
    }

    // The chip can move or vanish under a parked pointer: the editor scrolls,
    // the window resizes, or the user deletes the reference the pill points at.
    const reanchor = () => setAnchor(current => (current ? anchorFor(current.chip) : null))

    document.addEventListener('pointerover', onPointerOver)
    document.addEventListener('pointerout', onPointerOut)
    window.addEventListener('scroll', reanchor, true)
    window.addEventListener('resize', reanchor)

    return () => {
      document.removeEventListener('pointerover', onPointerOver)
      document.removeEventListener('pointerout', onPointerOut)
      window.removeEventListener('scroll', reanchor, true)
      window.removeEventListener('resize', reanchor)
      window.clearTimeout(hideTimerRef.current)
    }
  }, [cancelHide, editorRef, hideSoon])

  if (!anchor) {
    return null
  }

  return createPortal(
    <div
      className="fixed z-(--z-over-modal) -translate-y-full pb-1"
      data-slot="composer-directive-action"
      data-value={anchor.value}
      onMouseEnter={cancelHide}
      onMouseLeave={hideSoon}
      style={{ left: anchor.left, top: anchor.top }}
    >
      <button
        className={cn(composerFloatingPill, 'shadow-nous')}
        onClick={() => {
          anchor.action.run(anchor.value)
          setAnchor(null)
        }}
        // Never let the press reach the editor: mousedown inside a
        // contenteditable moves the caret and can collapse a selection the user
        // still wants, and the edit composer treats a blur as "cancel".
        onMouseDown={event => event.preventDefault()}
        type="button"
      >
        <Codicon className="shrink-0 opacity-70" name={anchor.action.icon} size="0.75rem" />
        {anchor.action.label(t)}
      </button>
    </div>,
    document.body
  )
}
