'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { addMessageSelectionToChat } from '@/app/chat/composer/selection-composer-bridge'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'

/**
 * Smallest delay (ms) between the selection becoming stable and the quote
 * button appearing. Avoids flicker during a normal drag-select.
 */
const QUOTE_DELAY_MS = 200

interface MessageQuoteButtonProps {
  /** Stable message id scoped to this chat message. */
  messageId: string
}

/**
 * A floating "Quote in chat" button that appears when the user selects text
 * inside the enclosing assistant or user message bubble.
 *
 * Renders as a `position:fixed` pill near the selection's top-center edge.
 * Clicking it calls {@link addMessageSelectionToChat} with the selected text
 * and the message id, then clears the selection.
 *
 * ### Scoping
 * Finds the nearest ancestor with `data-slot="aui_assistant-message-root"` or
 * `"aui_user-message-root"` — the same attributes the thread components stamp
 * on their outer wrapper. Only text selections anchored and focused inside
 * that subtree trigger the button.
 *
 * ### Lifecycle
 * - Appears ~200ms after a stable non-collapsed selection inside the message.
 * - Disappears when the selection collapses, moves outside the message, or
 *   the user clicks anywhere outside the button (dismiss).
 */
export function MessageQuoteButton({ messageId }: MessageQuoteButtonProps) {
  const { t } = useI18n()
  const [show, setShow] = useState(false)
  const [style, setStyle] = useState<React.CSSProperties>({})
  const anchorRef = useRef<HTMLSpanElement>(null)
  const btnRef = useRef<HTMLButtonElement>(null)
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const messageRootRef = useRef<HTMLElement | null>(null)
  const cleanupDismissRef = useRef<(() => void) | null>(null)

  const hide = useCallback(() => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current)
      hideTimerRef.current = null
    }
    setShow(false)
  }, [])

  const scheduleHide = useCallback(() => {
    if (hideTimerRef.current) return
    hideTimerRef.current = setTimeout(hide, QUOTE_DELAY_MS)
  }, [hide])

  useEffect(() => {
    const span = anchorRef.current
    if (!span) return

    // Find the enclosing message root once. The hidden <span> is always
    // mounted inside the message component's DOM subtree, so `closest`
    // resolves it immediately.
    messageRootRef.current = span.closest(
      '[data-slot="aui_assistant-message-root"], [data-slot="aui_user-message-root"]'
    ) as HTMLElement | null

    const root = messageRootRef.current
    if (!root) return

    const onSelectionChange = () => {
      const sel = window.getSelection()

      // No selection or collapsed — schedule a deferred hide so normal
      // drag-select doesn't flicker the button multiple times.
      if (!sel || sel.isCollapsed || !sel.toString().trim()) {
        scheduleHide()
        return
      }

      // Both anchor and focus must be inside this message.
      if (!root.contains(sel.anchorNode) || !root.contains(sel.focusNode)) {
        hide()
        return
      }

      // Cancel any pending hide — selection is valid and inside our bubble.
      if (hideTimerRef.current) {
        clearTimeout(hideTimerRef.current)
        hideTimerRef.current = null
      }

      // Position above the selection's top-center edge.
      const range = sel.getRangeAt(0)
      const rect = range.getBoundingClientRect()

      setStyle({
        position: 'fixed',
        top: `${rect.top - 8}px`,
        left: `${rect.left + rect.width / 2}px`,
        transform: 'translateX(-50%) translateY(-100%)'
      })
      setShow(true)
    }

    // Defer the pointerdown listener by a microtask so the click that
    // triggered the button's own handler doesn't immediately dismiss it.
    const dismissTimeout = setTimeout(() => {
      const onPointerDown = (e: MouseEvent) => {
        if (btnRef.current && !btnRef.current.contains(e.target as Node)) {
          hide()
        }
      }
      document.addEventListener('pointerdown', onPointerDown)
      cleanupDismissRef.current = () => document.removeEventListener('pointerdown', onPointerDown)
    }, 0)

    document.addEventListener('selectionchange', onSelectionChange)

    return () => {
      document.removeEventListener('selectionchange', onSelectionChange)
      cleanupDismissRef.current?.()
      cleanupDismissRef.current = null
      clearTimeout(dismissTimeout)
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current)
    }
  }, [scheduleHide, hide])

  const quote = useCallback(() => {
    const sel = window.getSelection()
    if (!sel || sel.isCollapsed) return

    const text = sel.toString().trim()
    if (!text) return

    triggerHaptic('selection')
    addMessageSelectionToChat(text, messageId)
    sel.removeAllRanges()
    hide()
  }, [messageId, hide])

  return (
    <>
      {/*
       * Always-mounted anchor used to locate the message root via DOM
       * traversal. Must be a child of the message component so `closest`
       * resolves to the correct root irrespective of show/hide state.
       */}
      <span ref={anchorRef} />
      {show && (
        <button
          ref={btnRef}
          className="fixed z-50 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-popover-background) px-2 py-1 text-xs font-medium text-foreground shadow-lg transition-colors hover:bg-accent active:scale-95"
          onClick={event => {
            event.preventDefault()
            event.stopPropagation()
            quote()
          }}
          style={style}
          type="button"
        >
          {t.assistant.thread.quoteInChat}
        </button>
      )}
    </>
  )
}