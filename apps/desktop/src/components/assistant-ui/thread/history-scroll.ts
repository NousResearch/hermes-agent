const cssEscape = (value: string): string => {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(value)
  }

  return value.replace(/[^a-zA-Z0-9_:-]/g, ch => `\\${ch}`)
}

/** Stable visible occurrences, not distance from a moving/evicted page edge. */
export interface HistoryScrollAnchor {
  key: string
  offset: number
  occurrence: number
}

export function captureHistoryScroll(viewport: HTMLElement): HistoryScrollAnchor[] {
  const bounds = viewport.getBoundingClientRect()
  const candidates: (HistoryScrollAnchor & { leaf: boolean })[] = []

  // Read outer containment boxes first; don't force every offscreen turn's
  // content-visibility subtree to lay out just to capture one reading position.
  for (const group of viewport.querySelectorAll<HTMLElement>('[data-slot="aui_message-group"]')) {
    const box = group.getBoundingClientRect()

    if (box.bottom <= bounds.top || box.top >= bounds.bottom) {
      continue
    }

    const occurrences = new Map<string, number>()

    for (const node of [group, ...group.querySelectorAll<HTMLElement>('[data-history-anchor]')]) {
      const key = node.dataset.historyAnchor

      if (!key) {
        continue
      }

      const occurrence = occurrences.get(key) ?? 0
      occurrences.set(key, occurrence + 1)
      const rect = node.getBoundingClientRect()

      if (!rect.height || rect.bottom <= bounds.top || rect.top >= bounds.bottom) {
        continue
      }

      candidates.push({ key, occurrence, offset: rect.top - bounds.top, leaf: node !== group })
    }
  }

  return candidates
    .sort((a, b) => Number(b.leaf) - Number(a.leaf) || Math.abs(a.offset) - Math.abs(b.offset))
    .map(({ key, offset, occurrence }) => ({ key, offset, occurrence }))
}

export function restoreHistoryScroll(viewport: HTMLElement, anchors: readonly HistoryScrollAnchor[]): boolean {
  for (const anchor of anchors) {
    const node = viewport.querySelectorAll<HTMLElement>(`[data-history-anchor="${cssEscape(anchor.key)}"]`)[
      anchor.occurrence
    ]

    if (!node || !node.getBoundingClientRect().height) {
      continue
    }

    viewport.scrollTop += node.getBoundingClientRect().top - viewport.getBoundingClientRect().top - anchor.offset

    return true
  }

  return false
}

/**
 * Runtime publication and descendant layout are separate commits. Keep the
 * chosen occurrence through later DOM/size changes, not an arbitrary number
 * of frames. The owner releases this hold on user navigation or scope change.
 * An idle historical page does no work: there is no polling/animation loop.
 */
export function holdHistoryScroll(
  viewport: HTMLElement,
  content: HTMLElement,
  anchors: readonly HistoryScrollAnchor[],
  beforeRestore: () => void
): () => void {
  let active = true
  let frame = 0

  const restore = () => {
    if (!active) {
      return
    }

    beforeRestore()
    restoreHistoryScroll(viewport, anchors)
  }

  const schedule = () => {
    if (active && !frame) {
      frame = requestAnimationFrame(() => {
        frame = 0
        restore()
      })
    }
  }

  const resize = new ResizeObserver(restore)
  const mutations = new MutationObserver(schedule)
  resize.observe(content)
  mutations.observe(content, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['data-message-id', 'data-history-anchor']
  })
  restore()
  schedule()

  return () => {
    active = false
    cancelAnimationFrame(frame)
    resize.disconnect()
    mutations.disconnect()
  }
}
