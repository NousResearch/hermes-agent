/**
 * The "chat about selection" pill.
 *
 * The same verb already lives in the transcript context menu; this is the
 * in-place entrance — select text, an action appears where the eyes already are.
 * It is deliberately a PILL rather than a second menu: right-click belongs to the
 * app/Electron menu, and a primary-button menu would fight it.
 *
 * Positioned in VIEWPORT coordinates and dismissed on scroll. The transcript is
 * virtualised, so an element anchored to a message outlives the node it was
 * measured from; a pill pinned to a recycled row is worse than no pill at all.
 */

import { type RefObject, useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'

import { Codicon } from '@/components/ui/codicon'
import { writeClipboardText } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { requestSideChat } from '@/store/side-chat'

const EDGE_PAD = 8
const PILL_HEIGHT = 32

interface Placement {
  left: number
  top: number
}

/** Marks the pill so its own mousedown/mouseup never dismisses it. */
const PILL_ATTR = 'data-selection-toolbar'

function insidePill(target: EventTarget | null): boolean {
  return target instanceof Element && Boolean(target.closest(`[${PILL_ATTR}]`))
}

export function SelectionToolbar({ container }: { container: RefObject<HTMLElement | null> }) {
  const { t } = useI18n()
  const [placement, setPlacement] = useState<null | Placement>(null)

  const [selection, setSelection] = useState<null | {
    fromStoredSessionId?: string
    messageId?: string
    text: string
  }>(null)

  const hide = useCallback(() => setPlacement(null), [])

  useEffect(() => {
    const onMouseUp = (event: MouseEvent) => {
      // The click that lands on the pill must reach it: this fires before click,
      // and dismissing here would unmount the button mid-gesture.
      if (event.button !== 0 || insidePill(event.target)) {
        return
      }

      const root = container.current
      const selection = window.getSelection()
      const selected = selection?.toString().trim() ?? ''

      if (!root || !selection?.rangeCount || !selected) {
        hide()

        return
      }

      const range = selection.getRangeAt(0)
      const anchor = range.commonAncestorContainer
      const element = anchor.nodeType === Node.ELEMENT_NODE ? (anchor as Element) : anchor.parentElement
      // Attribute from the START of the range, not from its common ancestor: a
      // selection spanning rows has the message LIST as its common ancestor,
      // which belongs to no message — provenance would go missing exactly when
      // the user quoted most.
      const start = range.startContainer
      const startElement = start.nodeType === Node.ELEMENT_NODE ? (start as Element) : start.parentElement

      // Only THIS transcript's selections — the app mounts several chat surfaces
      // (tiles) plus all the chrome, and those keep their own verbs.
      if (!element || !root.contains(element)) {
        hide()

        return
      }

      // A selection inside a field is an EDIT gesture (rewriting a message), not
      // a request to discuss it — the transcript hosts those editors itself.
      if (startElement?.closest('input, textarea, [contenteditable]')) {
        hide()

        return
      }

      const rect = range.getBoundingClientRect()

      if (!rect.width && !rect.height) {
        hide()

        return
      }

      // A selection spanning rows attributes to the FIRST row holding any of it.
      // The transcript names its own conversation, so the request carries it
      // rather than letting the create path fall back to whatever session the
      // app considers selected (which lags a route switch).
      setSelection({
        fromStoredSessionId: root.getAttribute('data-stored-session-id') || undefined,
        messageId: startElement?.closest('[data-message-id]')?.getAttribute('data-message-id') ?? undefined,
        text: selected
      })
      setPlacement({
        left: Math.max(EDGE_PAD, Math.min(rect.left + rect.width / 2, window.innerWidth - EDGE_PAD)),
        top: rect.top - PILL_HEIGHT - EDGE_PAD < EDGE_PAD ? rect.bottom + EDGE_PAD : rect.top - PILL_HEIGHT - EDGE_PAD
      })
    }

    const onMouseDown = (event: MouseEvent) => {
      if (!insidePill(event.target)) {
        hide()
      }
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        hide()
      }
    }

    document.addEventListener('mouseup', onMouseUp)
    document.addEventListener('mousedown', onMouseDown)
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('scroll', hide, true)

    return () => {
      document.removeEventListener('mouseup', onMouseUp)
      document.removeEventListener('mousedown', onMouseDown)
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('scroll', hide, true)
    }
  }, [container, hide])

  if (!placement || !selection) {
    return null
  }

  // Portalled to the body: the transcript's own subtree sits inside
  // `contain-[layout_paint]` (the pane's paint isolation), and `contain: paint`
  // makes that element a containing block for fixed-position descendants — a
  // pill left in place would be laid out against the pane corner while its
  // coordinates come from the viewport.
  return createPortal(
    <div
      className="fixed z-50 flex -translate-x-1/2 items-center gap-0.5 rounded-full border border-(--stroke-nous) bg-popover/95 p-0.5 shadow-nous backdrop-blur-md"
      data-selection-toolbar=""
      style={{ left: placement.left, top: placement.top }}
    >
      <button
        className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.8125rem] whitespace-nowrap hover:bg-accent"
        onClick={() => {
          hide()
          requestSideChat(selection)
        }}
        type="button"
      >
        <Codicon name="comment-discussion" size="0.875rem" />
        <span>{t.desktop.sideChat.chatAboutSelection}</span>
      </button>
      <button
        aria-label={t.common.copy}
        className="grid size-6 place-items-center rounded-full text-muted-foreground hover:bg-accent hover:text-foreground"
        onClick={() => {
          void writeClipboardText(selection.text)
          hide()
        }}
        type="button"
      >
        <Codicon name="copy" size="0.75rem" />
      </button>
    </div>,
    document.body
  )
}
