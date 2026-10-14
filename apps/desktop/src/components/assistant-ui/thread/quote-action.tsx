import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'

import { captureQuoteSelection, quoteBlock, type QuoteCapture, selectionAnchorRect } from './quote-selection'

/** Distance between the selection and the action, in px. */
const GAP = 8

/** Below this the action would clip off the top of the window, so it flips under
 *  the selection instead (a quote near the first line of a scrolled transcript). */
const FLIP_BELOW_UNDER = 40

interface Anchor {
  left: number
  top: number
  flipped: boolean
}

/**
 * The floating "Add to chat" action for a transcript selection.
 *
 * Rendered once per transcript pane, portaled to the body so no scrolling or
 * clipping ancestor can crop it, and positioned from the selection's own rect.
 * Appearing NEVER steals focus or the selection: the button opens on
 * mousedown-preventDefault (the gesture the terminal pane's own Add-to-chat uses)
 * so the browser's selection survives the click that quotes it.
 */
export function QuoteAction() {
  const { t } = useI18n()
  const [anchor, setAnchor] = useState<Anchor | null>(null)
  const captureRef = useRef<QuoteCapture | null>(null)

  const dismiss = useCallback(() => {
    captureRef.current = null
    setAnchor(null)
  }, [])

  const sync = useCallback(() => {
    const capture = captureQuoteSelection(window.getSelection())
    const rect = capture ? selectionAnchorRect(window.getSelection()) : null

    if (!capture || !rect) {
      dismiss()

      return
    }

    captureRef.current = capture

    const flipped = rect.top < FLIP_BELOW_UNDER

    setAnchor({
      left: rect.left + rect.width / 2,
      top: flipped ? rect.bottom + GAP : rect.top - GAP,
      flipped
    })
  }, [dismiss])

  useEffect(() => {
    // selectionchange fires on every pixel of a drag; coalesce to one frame. The
    // frame id stays local to the effect — nothing outside it cancels a frame.
    let frame = 0

    const schedule = () => {
      if (frame) {
        return
      }

      frame = window.requestAnimationFrame(() => {
        frame = 0
        sync()
      })
    }

    // A scroll under a live selection moves the anchor without firing
    // selectionchange, so the scroll pass is captured (any pane's scroll).
    document.addEventListener('selectionchange', schedule)
    window.addEventListener('scroll', schedule, true)
    window.addEventListener('resize', schedule)

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        dismiss()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => {
      document.removeEventListener('selectionchange', schedule)
      window.removeEventListener('scroll', schedule, true)
      window.removeEventListener('resize', schedule)
      window.removeEventListener('keydown', onKeyDown)

      if (frame) {
        window.cancelAnimationFrame(frame)
      }
    }
  }, [dismiss, sync])

  const quote = useCallback(() => {
    const capture = captureRef.current

    if (!capture) {
      return
    }

    requestComposerInsert(quoteBlock(capture), { caret: 'below', mode: 'block' })
    requestComposerFocus('active')
    window.getSelection()?.removeAllRanges()
    dismiss()
  }, [dismiss])

  if (!anchor) {
    return null
  }

  return createPortal(
    <div
      className="fixed z-50 -translate-x-1/2"
      style={{
        left: anchor.left,
        top: anchor.top,
        transform: anchor.flipped ? 'translate(-50%, 0)' : 'translate(-50%, -100%)'
      }}
    >
      <Button
        className="h-6 rounded-md px-2 text-[0.68rem]"
        onClick={event => event.preventDefault()}
        onMouseDown={event => {
          // Keep the browser selection alive through the click that quotes it —
          // and never let the press reach the transcript underneath.
          event.preventDefault()
          event.stopPropagation()
          quote()
        }}
        type="button"
        variant="floating"
      >
        {t.composer.quoteAddToChat}
      </Button>
    </div>,
    document.body
  )
}
