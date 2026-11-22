import { type ReactNode, type RefObject, useLayoutEffect, useRef } from 'react'

const GROUP = '[data-slot="aui_message-group"]'
const PROMPT = '[data-slot="aui_user-message-root"]'
// Registered as a non-inherited <length> in styles.css. That registration is
// load-bearing: this is written once per scroll frame onto elements that are
// whole turns, and an unregistered property forces Blink to restyle every
// descendant on each write. See the @property block there before changing it.
const CLIP = '--sticky-prompt-clip'

/**
 * Dispatched synchronously on the scroll viewport by code that writes its
 * scrollTop from a later ResizeObserver (bottom-follow while streaming). The
 * scroll event would only re-measure on the next frame, after a stale paint.
 */
export const STICKY_PROMPT_CLIP_REMEASURE = 'sticky-prompt-clip:remeasure'

interface StickyPromptClipOptions {
  contentRef: RefObject<HTMLElement | null>
  scrollRef: RefObject<HTMLElement | null>
  paneVisible: boolean
  rows: ReactNode
}

/** Clip covered content instead of painting a solid rectangle over the glass. */
export function useStickyPromptClip({ contentRef, scrollRef, paneVisible, rows }: StickyPromptClipOptions) {
  const controller = useRef<ReturnType<typeof observeStickyPromptClip> | null>(null)

  useLayoutEffect(() => {
    const viewport = scrollRef.current
    const content = contentRef.current
    const previous = controller.current

    if (!paneVisible || previous?.viewport !== viewport || previous?.content !== content) {
      previous?.dispose()
      controller.current = null
    }

    if (paneVisible && viewport && content) {
      controller.current ??= observeStickyPromptClip(viewport, content)
      // Reconcile before paint without tearing down surviving clips when a
      // stream appends a message or the history window changes its rows.
      controller.current.reconcile()
    }
  }, [contentRef, scrollRef, paneVisible, rows])

  useLayoutEffect(
    () => () => {
      controller.current?.dispose()
      controller.current = null
    },
    []
  )
}

function observeStickyPromptClip(viewport: HTMLElement, content: HTMLElement) {
  const observed = new Set<HTMLElement>()
  const visible = new Set<HTMLElement>()
  const clipped = new Set<HTMLElement>()
  let frame = 0

  const measure = () => {
    frame = 0
    const viewportTop = viewport.getBoundingClientRect().top
    const next = new Map<HTMLElement, number>()
    let activePrompt: HTMLElement | null = null
    let exclusionBottom = viewportTop

    for (const group of visible) {
      // Read only intersecting groups: measuring a skipped turn's descendants
      // would defeat content-visibility and wake the entire transcript.
      const prompt = group.querySelector<HTMLElement>(PROMPT)

      if (!prompt) {
        continue
      }

      // Only a pinned (sticky) prompt owns an exclusion. Process notifications
      // and agent deliveries share the prompt's data-slot (and its `top` inset
      // from styles.css) but scroll in flow. Taken as the "later prompt" while
      // crossing the top edge, one stretched the clip from the real card down
      // to its own bottom: an opaque band over the transcript below the card.
      const promptStyle = getComputedStyle(prompt)

      if (promptStyle.position !== 'sticky') {
        continue
      }

      const stickyTop = Number.parseFloat(promptStyle.top) || 0
      const promptRect = prompt.getBoundingClientRect()

      if (promptRect.top > viewportTop + stickyTop + 1 || promptRect.bottom <= viewportTop) {
        continue
      }

      // During the handoff, the later prompt supersedes the one being pushed
      // out. Its exclusion region also covers preceding turns in the top gap.
      if (!activePrompt || activePrompt.compareDocumentPosition(prompt) & Node.DOCUMENT_POSITION_FOLLOWING) {
        activePrompt = prompt
        exclusionBottom = promptRect.bottom
      }
    }

    if (activePrompt) {
      // Follow only the ancestor path to the active prompt. Other groups can
      // be clipped whole, including old prompts and standalone assistant rows.
      const collect = (element: HTMLElement) => {
        if (element === activePrompt) {
          return
        }

        if (element.contains(activePrompt)) {
          for (const child of element.children) {
            if (child instanceof HTMLElement) {
              collect(child)
            }
          }

          return
        }

        const rect = element.getBoundingClientRect()

        if (rect.height > 0 && rect.top < exclusionBottom) {
          next.set(element, Math.min(rect.height, exclusionBottom - rect.top))
        }
      }

      for (const group of visible) {
        // Keep the IO target itself unmasked; clipping it would change its
        // intersection and oscillate between hiding and revealing the group.
        for (const child of group.children) {
          if (child instanceof HTMLElement) {
            collect(child)
          }
        }
      }
    }

    // All geometry reads precede writes. Clipping changes no layout, scroll
    // position, or React state; only covered siblings get a style update.
    for (const element of clipped) {
      if (!next.has(element)) {
        element.style.removeProperty(CLIP)
        element.removeAttribute('data-sticky-prompt-clip')
      }
    }

    clipped.clear()

    for (const [element, inset] of next) {
      const value = `${inset}px`

      if (element.style.getPropertyValue(CLIP) !== value) {
        element.style.setProperty(CLIP, value)
        element.setAttribute('data-sticky-prompt-clip', '')
      }

      clipped.add(element)
    }
  }

  const schedule = () => {
    if (!frame) {
      frame = requestAnimationFrame(measure)
    }
  }

  // Outer boxes only: never force layout inside content-visibility's skipped rows.
  const refreshVisible = () => {
    const viewportRect = viewport.getBoundingClientRect()

    for (const group of observed) {
      const rect = group.getBoundingClientRect()

      if (rect.bottom > viewportRect.top && rect.top < viewportRect.bottom) {
        visible.add(group)
        sizes.observe(group)
      } else {
        visible.delete(group)
        sizes.unobserve(group)
      }
    }
  }

  // Size changes are delivered after layout and before paint, in the frame
  // whose geometry they describe. Measuring there (not one rAF later) matters
  // when the viewport itself resizes: the browser clamps scrollTop, the
  // content moves under a clip still sized for the old position, and for one
  // painted frame a band below the pinned card hid the transcript. Any resize
  // (viewport, a collapsing row, a clamped scrollTop) can also bring a turn
  // into view before IntersectionObserver delivers it, so refresh visibility
  // from outer boxes first.
  const sizes: ResizeObserver = new ResizeObserver(() => {
    cancelAnimationFrame(frame)
    refreshVisible()
    measure()
  })

  const intersections = new IntersectionObserver(
    entries => {
      for (const entry of entries) {
        const group = entry.target as HTMLElement

        if (entry.isIntersecting) {
          visible.add(group)
          sizes.observe(group)
        } else {
          visible.delete(group)
          sizes.unobserve(group)
        }
      }

      schedule()
    },
    { root: viewport }
  )

  const reconcile = () => {
    const groups = new Set(content.querySelectorAll<HTMLElement>(GROUP))
    const viewportRect = viewport.getBoundingClientRect()

    for (const group of observed) {
      if (!groups.has(group)) {
        intersections.unobserve(group)
        sizes.unobserve(group)
        observed.delete(group)
        visible.delete(group)
      }
    }

    for (const group of groups) {
      if (!observed.has(group)) {
        intersections.observe(group)
        observed.add(group)
      }

      // A row commit can rehome a turn before IO delivers. Measure only outer
      // boxes here; never force layout inside content-visibility's skipped rows.
      const rect = group.getBoundingClientRect()

      if (rect.bottom > viewportRect.top && rect.top < viewportRect.bottom) {
        visible.add(group)
        sizes.observe(group)
      } else {
        visible.delete(group)
        sizes.unobserve(group)
      }
    }

    cancelAnimationFrame(frame)
    measure()
  }

  const remeasure = () => {
    cancelAnimationFrame(frame)
    refreshVisible()
    measure()
  }

  sizes.observe(viewport)
  sizes.observe(content)
  viewport.addEventListener('scroll', schedule, { passive: true })
  viewport.addEventListener(STICKY_PROMPT_CLIP_REMEASURE, remeasure)

  const dispose = () => {
    intersections.disconnect()
    sizes.disconnect()
    viewport.removeEventListener('scroll', schedule)
    viewport.removeEventListener(STICKY_PROMPT_CLIP_REMEASURE, remeasure)
    cancelAnimationFrame(frame)

    for (const element of clipped) {
      element.style.removeProperty(CLIP)
      element.removeAttribute('data-sticky-prompt-clip')
    }
  }

  return { viewport, content, reconcile, dispose }
}
