import { type RefObject, useLayoutEffect, useMemo, useRef } from 'react'

const GROUP = '[data-slot="aui_message-group"]'
const PROMPT = '[data-slot="aui_user-message-root"]'
// Registered as a non-inherited <length> in styles.css. That registration is
// load-bearing: this is written once per scroll frame onto elements that are
// whole turns, and an unregistered property forces Blink to restyle every
// descendant on each write. See the @property block there before changing it.
const CLIP = '--sticky-prompt-clip'

interface StickyPromptClipOptions {
  contentRef: RefObject<HTMLElement | null>
  scrollRef: RefObject<HTMLElement | null>
  paneVisible: boolean
}

/** Clip covered content instead of painting a solid rectangle over the glass. */
export function useStickyPromptClip({ contentRef, scrollRef, paneVisible }: StickyPromptClipOptions) {
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
      // Full pass only on pane/viewport transitions. Row membership changes are
      // handled by register()/unregister() from each row's own ref — the old
      // per-rows reconcile() walked every mounted group (querySelectorAll +
      // getBoundingClientRect) on every backfill step, paging commit, and
      // stream append, which scaled the cost of each commit with the size of
      // the whole mounted transcript.
      controller.current.reconcile()
    }
  }, [contentRef, scrollRef, paneVisible])

  // Stable registration surface for row refs (see TurnRow). The closures read
  // controller.current at call time, so this object never changes identity
  // even when the controller is recreated on a pane/viewport swap; a swapped
  // controller repopulates itself through reconcile() because row effects
  // don't re-run on the swap.
  const rowClip = useMemo(
    () => ({
      register: (el: HTMLElement) => controller.current?.register(el),
      unregister: (el: HTMLElement) => controller.current?.unregister(el)
    }),
    []
  )

  useLayoutEffect(
    () => () => {
      controller.current?.dispose()
      controller.current = null
    },
    []
  )

  return rowClip
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

      const promptRect = prompt.getBoundingClientRect()
      const stickyTop = Number.parseFloat(getComputedStyle(prompt).top) || 0

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

  const sizes = new ResizeObserver(schedule)

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

  sizes.observe(viewport)
  sizes.observe(content)
  viewport.addEventListener('scroll', schedule, { passive: true })

  // Incremental membership: each TurnRow root registers itself from its own
  // ref on mount and unregisters on unmount. This replaces the per-rows
  // reconcile() diff walk (querySelectorAll over every mounted group plus a
  // bounding check per group on every rows-identity change) with O(1) work per
  // row transition. IO still delivers real visibility asynchronously; the same
  // outer-box pre-warm reconcile used keeps a rehomed turn correct before the
  // first IO delivery.
  const register = (group: HTMLElement) => {
    if (observed.has(group)) {
      return
    }

    intersections.observe(group)
    observed.add(group)

    const rect = group.getBoundingClientRect()
    const viewportRect = viewport.getBoundingClientRect()

    if (rect.bottom > viewportRect.top && rect.top < viewportRect.bottom) {
      visible.add(group)
      sizes.observe(group)
    }

    schedule()
  }

  const unregister = (group: HTMLElement) => {
    if (!observed.has(group)) {
      return
    }

    intersections.unobserve(group)
    sizes.unobserve(group)
    observed.delete(group)
    visible.delete(group)
    schedule()
  }

  const dispose = () => {
    intersections.disconnect()
    sizes.disconnect()
    viewport.removeEventListener('scroll', schedule)
    cancelAnimationFrame(frame)

    for (const element of clipped) {
      element.style.removeProperty(CLIP)
      element.removeAttribute('data-sticky-prompt-clip')
    }
  }

  return { viewport, content, reconcile, register, unregister, dispose }
}
