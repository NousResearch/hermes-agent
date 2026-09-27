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
    const node = viewport.querySelectorAll<HTMLElement>(`[data-history-anchor="${CSS.escape(anchor.key)}"]`)[
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
