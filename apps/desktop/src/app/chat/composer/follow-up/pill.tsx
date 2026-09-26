import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { requestComposerFocus } from '@/app/chat/composer/focus'
import { useComposerScope } from '@/app/chat/composer/scope'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { KbdCombo } from '@/components/ui/kbd'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { isComposerChord } from '@/lib/keybinds/chords'
import { OVERLAY_SURFACE } from '@/lib/keybinds/combo'

import { captureFollowUpSelection, type FollowUpCapture } from './passage'

/** Clearance between the passage and the pill's bottom edge. */
const PILL_OFFSET_PX = 8
/** Smallest margin a clamped pill keeps from the transcript's edges. */
const PILL_MARGIN_PX = 8

interface Placement {
  capture: FollowUpCapture
  /** Selection center in host coordinates (the pill centers on it). */
  left: number
  /** Selection top in host coordinates (the pill sits above it). */
  top: number
}

interface FollowUpPillProps {
  /** The transcript's scroll viewport: a quotable selection has to live here. */
  viewportRef: { current: HTMLElement | null }
}

/**
 * The transcript's "Follow-up" affordance: pick a passage out of an answer and
 * it becomes the quote the next message answers.
 *
 * Same shape as the terminal's "Add to chat" pill (`right-sidebar/terminal`):
 * one control that appears over the selection it would quote, and the ⌘/Ctrl+L
 * chord as its keyboard half. It differs in where the capture GOES — a
 * terminal selection rides an `@terminal:` ref in the draft, a transcript
 * passage rides the composer's follow-up card (see `store/composer`) — because
 * a quoted passage has nothing to address, only text to carry.
 *
 * Everything here is transient: the pill's own markup is the only thing it
 * owns, and the passage itself lives in the composer's scope the moment the
 * user commits to it.
 */
export function FollowUpPill({ viewportRef }: FollowUpPillProps) {
  const scope = useComposerScope()
  const { t } = useI18n()
  const hostRef = useRef<HTMLDivElement>(null)
  const pillRef = useRef<HTMLButtonElement>(null)
  const [placement, setPlacement] = useState<Placement | null>(null)

  const read = useCallback((): Placement | null => {
    const viewport = viewportRef.current
    const host = hostRef.current

    if (!viewport || !host) {
      return null
    }

    const selection = document.getSelection()
    const capture = captureFollowUpSelection(selection, viewport)

    if (!capture || !selection || selection.rangeCount === 0) {
      return null
    }

    // A popover owns the window: the tapback picker opens on the same
    // double-click that selects a word (use-message-reactions), and a second
    // floating surface stacked on it reads as a bug.
    if (document.querySelector(OVERLAY_SURFACE)) {
      return null
    }

    const rect = selection.getRangeAt(0).getBoundingClientRect()

    if (!rect.width && !rect.height) {
      return null
    }

    const hostRect = host.getBoundingClientRect()

    return {
      capture,
      left: rect.left + rect.width / 2 - hostRect.left,
      top: rect.top - hostRect.top
    }
  }, [viewportRef])

  // Identity-preserving publish: a drag fires selectionchange dozens of times
  // and the same placement must not re-render the pill each of them.
  const publish = useCallback((next: Placement | null) => {
    setPlacement(current => {
      if (!next || !current) {
        return current === next ? current : next
      }

      return current.left === next.left &&
        current.top === next.top &&
        current.capture.passage === next.capture.passage &&
        current.capture.source === next.capture.source
        ? current
        : next
    })
  }, [])

  const attach = useCallback(
    (capture: FollowUpCapture) => {
      scope.followUp.set(capture)
      // The passage lives in the card above the composer now, so releasing the
      // DOM selection both dismisses the pill and keeps "what is selected" from
      // disagreeing with "what is attached".
      document.getSelection()?.removeAllRanges()
      setPlacement(null)
      triggerHaptic('selection')
      // The scope's target, never 'active': a tile's passage belongs in the
      // tile's composer even when another composer was focused last.
      requestComposerFocus(scope.target)
    },
    [scope]
  )

  useEffect(() => {
    // The pill is offered when the gesture ENDS, never while a drag is still
    // choosing text: a control that tracks the pointer mid-selection fights the
    // selection itself, and lands under the cursor that is still working.
    const finish = () => publish(read())
    const dismiss = () => publish(null)

    const beginGesture = (event: PointerEvent) => {
      // A press ON the pill is the click that takes the offer, not a new
      // selection — retiring it here would unmount the button mid-press.
      // (`target` can be a non-Element for synthetic events: no containment to
      // test, so let the press through.)
      const target = event.target

      if (pillRef.current && target instanceof Node && pillRef.current.contains(target)) {
        return
      }

      dismiss()
    }

    // The selection is gone (Escape, a collapse somewhere else, a programmatic
    // clear): drop the offer without publishing a new one.
    const selectionLost = () => {
      if (!read()) {
        dismiss()
      }
    }

    // Capture phase: a drag that ends on a control inside the transcript still
    // has to re-read the selection, and the transcript's own handlers stop
    // propagation on some of those.
    window.addEventListener('pointerdown', beginGesture, true)
    window.addEventListener('pointerup', finish, true)
    window.addEventListener('keyup', finish, true)
    document.addEventListener('selectionchange', selectionLost)

    // A scrolled transcript moves the text out from under the pill's rect, and
    // the selection itself does not move with it: dismiss rather than chase.
    const viewport = viewportRef.current
    viewport?.addEventListener('scroll', dismiss, { passive: true })

    return () => {
      window.removeEventListener('pointerdown', beginGesture, true)
      window.removeEventListener('pointerup', finish, true)
      window.removeEventListener('keyup', finish, true)
      document.removeEventListener('selectionchange', selectionLost)
      viewport?.removeEventListener('scroll', dismiss)
    }
  }, [publish, read, viewportRef])

  // The chord's selection half (the ladder lives in composer/focus-chord.ts):
  // with a live passage selected, ⌘/Ctrl+L sends it to the composer instead of
  // moving focus there.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!isComposerChord(event)) {
        return
      }

      const capture = captureFollowUpSelection(document.getSelection(), viewportRef.current)

      if (!capture) {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      attach(capture)
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [attach, viewportRef])

  // Clamp once the label has a width: a selection can end at either edge of the
  // transcript, and the transcript clips what leaves it. useLayoutEffect so the
  // clamp lands in the same frame as the pill's first paint.
  useLayoutEffect(() => {
    const host = hostRef.current
    const pill = pillRef.current

    if (!placement || !host || !pill) {
      return
    }

    const width = pill.offsetWidth
    const maxLeft = Math.max(PILL_MARGIN_PX, host.clientWidth - width - PILL_MARGIN_PX)
    const clamped = Math.min(Math.max(placement.left - (width ? width / 2 : 0), PILL_MARGIN_PX), maxLeft)
    const left = width ? clamped + width / 2 : placement.left

    if (left !== placement.left) {
      setPlacement({ ...placement, left })
    }
  }, [placement])

  return (
    <div className="pointer-events-none absolute inset-0 z-20" data-slot="aui_follow-up-pill-host" ref={hostRef}>
      {placement && (
        <Button
          className="pointer-events-auto absolute h-6 rounded-md px-2 text-[0.68rem] shadow-md backdrop-blur-md"
          data-slot="aui_follow-up-pill"
          onMouseDown={event => {
            // The press must not collapse the DOM selection before the passage
            // is read, and it must not hand focus to the composer behind the
            // click — attach() does both, in that order.
            event.preventDefault()
            event.stopPropagation()
            attach(placement.capture)
          }}
          ref={pillRef}
          style={{
            left: placement.left,
            top: placement.top,
            transform: `translate(-50%, calc(-100% - ${PILL_OFFSET_PX}px))`
          }}
          type="button"
          variant="floating"
        >
          <Codicon name="quote" />
          {t.composer.followUp.action}
          <KbdCombo combo="mod+l" size="sm" />
        </Button>
      )}
    </div>
  )
}
