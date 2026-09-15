import { type RefObject, useLayoutEffect, useRef } from 'react'

import { $turnAnchor, threadScrollStateFromMetrics, turnEndScrollTop } from '@/store/thread-scroll'

const GROUP = '[data-slot="aui_message-group"]'
const PROMPT = '[data-slot="aui_user-message-root"]'

/**
 * Viewport-relative top of the newest prompt's LAYOUT position, or null when
 * the transcript holds no prompt at all.
 *
 * The turn's group wrapper is measured, not the bubble: the bubble is
 * `position: sticky`, so once it has stuck to the viewport top its own rect
 * reports the PINNED position — which reads as "already at the top" for
 * exactly the long turn this anchor exists for.
 *
 * The last group HOLDING a prompt, not the last group: a turn can be followed
 * by standalone rows (a system notice, an injected note) that carry no prompt.
 */
export function newestPromptTop(content: HTMLElement): null | number {
  const groups = content.querySelectorAll<HTMLElement>(GROUP)

  for (let index = groups.length - 1; index >= 0; index -= 1) {
    const group = groups[index]!

    if (group.querySelector(PROMPT)) {
      return group.getBoundingClientRect().top
    }
  }

  return null
}

interface TurnEndAnchorOptions {
  contentRef: RefObject<HTMLElement | null>
  /** The run whose end this is (`useAuiState(s => s.thread.isRunning)`). */
  isRunning: boolean
  /** Only a settled load has a position the reader chose; a mid-load scrollTop
   *  is a way-point of the load itself (see loadSettledRef in list.tsx). */
  loadSettledRef: RefObject<boolean>
  scrollRef: RefObject<HTMLElement | null>
  stopScroll: () => void
}

/**
 * Settle the transcript where the reader asked for when a turn ends (#108941).
 *
 * Every guard is a deliberate hold: the default 'bottom' anchor keeps today's
 * landing untouched, a reader who scrolled up mid-turn keeps the position they
 * chose, and a transcript that is still loading stays where the load put it.
 */
export function useTurnEndAnchor({
  contentRef,
  isRunning,
  loadSettledRef,
  scrollRef,
  stopScroll
}: TurnEndAnchorOptions) {
  // A transition TOKEN, not a mirror of `isRunning`: it is read only by the
  // effect below, in the same commit that writes it, to tell an end-of-run from
  // the middle of one. Nothing outside reads it, so it cannot go stale.
  const wasRunningRef = useRef(isRunning)

  // Layout effect, deliberately: the end of a run arrives in the commit that
  // carries the answer's closing message, and the settle must land in THAT
  // frame — a passive effect (or a frame scheduled from the run-end event, which
  // the runtime fires before it publishes that message) paints the finished turn
  // at the bottom first and then jumps.
  useLayoutEffect(() => {
    const wasRunning = wasRunningRef.current

    wasRunningRef.current = isRunning

    if (!wasRunning || isRunning || !loadSettledRef.current) {
      return
    }

    const anchor = $turnAnchor.get()
    const viewport = scrollRef.current
    const content = contentRef.current

    if (anchor !== 'prompt' || !viewport || !content) {
      return
    }

    const target = turnEndScrollTop({
      anchor,
      atBottom: threadScrollStateFromMetrics(viewport).kind === 'bottom',
      maxScrollTop: viewport.scrollHeight - viewport.clientHeight,
      promptTop: newestPromptTop(content),
      scrollTop: viewport.scrollTop,
      viewportTop: viewport.getBoundingClientRect().top
    })

    if (target === null) {
      return
    }

    // Escape the follow lock BEFORE writing scrollTop: use-stick-to-bottom
    // re-pins on a resize event, and late layout (deferred markdown, images,
    // highlighting) can land one between the two writes.
    stopScroll()
    viewport.scrollTop = target
  }, [contentRef, isRunning, loadSettledRef, scrollRef, stopScroll])
}
