/**
 * Map a mouse text-selection inside the windowed Shiki source view to file
 * line numbers, for the "send selection to chat" context menu.
 *
 * Each 200-line chunk renders one `.preview-source-code` wrapper carrying its
 * 0-based `data-chunk-start`; Shiki emits one `.line` span per line under a
 * `<code>`. A boundary point (selection start/end) resolves to its `.line`,
 * whose absolute line number is `chunkStart + indexOfLineWithinChunk + 1`.
 */
export interface SourceLineRange {
  end: number
  start: number
}

function lineElementFrom(node: Node): HTMLElement | null {
  if (node.nodeType === Node.TEXT_NODE) {
    return node.parentElement?.closest('.line') ?? null
  }

  if (node instanceof Element) {
    return node.closest('.line')
  }

  return null
}

/** 1-based file line number of a rendered `.line` span, or null when it sits
 *  outside a chunked code wrapper. */
export function absoluteLineOf(lineEl: Element): null | number {
  const wrapper = lineEl.closest('[data-chunk-start]')
  const code = lineEl.parentElement

  if (!wrapper || !code) {
    return null
  }

  const base = Number(wrapper.getAttribute('data-chunk-start'))
  const index = Array.from(code.children).indexOf(lineEl)

  if (!Number.isFinite(base) || index < 0) {
    return null
  }

  return base + index + 1
}

/**
 * Resolve the `[start, end]` file line range (1-based, inclusive) a DOM
 * selection covers inside the source view's grid `container`. Returns null for
 * a collapsed selection, one outside the container, or one whose boundaries
 * don't land on a rendered line.
 *
 * A selection ending exactly at a line's start (caret parked there before any
 * character) hasn't selected content on that line, so the end is pulled back
 * one — dragging from line 5 to the very start of line 8 yields 5–7.
 */
export function sourceSelectionLineRange(container: HTMLElement, range: Range): null | SourceLineRange {
  if (range.collapsed || !container.contains(range.commonAncestorContainer)) {
    return null
  }

  const startLine = absoluteLineOf(lineElementFrom(range.startContainer) as Element)
  const endLine = absoluteLineOf(lineElementFrom(range.endContainer) as Element)

  if (startLine == null || endLine == null) {
    return null
  }

  let start = Math.min(startLine, endLine)
  let end = Math.max(startLine, endLine)

  if (end > start && range.endOffset === 0 && range.endContainer.nodeType === Node.TEXT_NODE) {
    end -= 1
  }

  return { end, start }
}
