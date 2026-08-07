import { droppedFileInlineRef } from '@/app/chat/composer/inline-refs'

export interface LineSelection {
  end: number
  start: number
}

/** `@line:path:start-end` chip for a gutter or resolved text selection in the file preview. */
export function sourceLineSelectionRef(
  filePath: string,
  selection: LineSelection,
  cwd: string | null | undefined
): null | string {
  const lineEnd = selection.end > selection.start ? selection.end : undefined

  return droppedFileInlineRef({ line: selection.start, lineEnd, path: filePath }, cwd)
}

/** Basename label for preview quotes when a line range cannot be resolved. */
export function previewSelectionFileLabel(filePath: string): string {
  return filePath.split(/[\\/]/).filter(Boolean).pop() || filePath.trim()
}

function normalizeNewlines(text: string) {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
}

/** 1-based inclusive line span covering [start, end) character offsets. */
export function lineSelectionFromOffsets(fullText: string, start: number, end: number): LineSelection | null {
  const text = normalizeNewlines(fullText)

  if (!text || start < 0 || end < start) {
    return null
  }

  const clampedStart = Math.min(start, text.length)
  const clampedEnd = Math.min(Math.max(end, clampedStart + 1), text.length + 1)
  const startLine = text.slice(0, clampedStart).split('\n').length
  const endOffset = Math.max(clampedStart, clampedEnd - 1)
  const endLine = text.slice(0, endOffset).split('\n').length

  return { end: endLine, start: startLine }
}

/**
 * Resolve a free-text selection to source line numbers by locating the selected
 * string in the file. Prefers an occurrence near `preferOffset` when the needle
 * appears more than once (virtualized source views only mount a window).
 */
export function lineSelectionFromSelectedText(
  fullText: string,
  selected: string,
  preferOffset?: number
): LineSelection | null {
  const haystack = normalizeNewlines(fullText)
  const needle = normalizeNewlines(selected)

  if (!haystack || !needle.trim()) {
    return null
  }

  const candidates: number[] = []
  let from = 0

  while (from <= haystack.length) {
    const idx = haystack.indexOf(needle, from)

    if (idx < 0) {
      break
    }

    candidates.push(idx)
    from = idx + Math.max(1, needle.length)
  }

  if (candidates.length === 0) {
    const trimmed = needle.trim()

    if (trimmed === needle) {
      return null
    }

    return lineSelectionFromSelectedText(haystack, trimmed, preferOffset)
  }

  let idx = candidates[0]!

  if (preferOffset != null && candidates.length > 1) {
    idx = candidates.reduce((best, next) =>
      Math.abs(next - preferOffset) < Math.abs(best - preferOffset) ? next : best
    )
  }

  return lineSelectionFromOffsets(haystack, idx, idx + needle.length)
}

/** Marks the preview Add-to-Chat frame so the terminal can defer ⌘/Ctrl+L. */
export const PREVIEW_ADD_TO_CHAT_ATTR = 'data-preview-add-to-chat'

function nodeInsideHost(host: HTMLElement, node: Node): boolean {
  return node === host || host.contains(node)
}

/** Clip `range` to `host` contents when a drag ends outside the frame. */
function clipRangeToHost(host: HTMLElement, range: Range): Range | null {
  const startIn = nodeInsideHost(host, range.startContainer)
  const endIn = nodeInsideHost(host, range.endContainer)

  if (startIn && endIn) {
    return range
  }

  if (!startIn && !endIn) {
    try {
      if (!range.intersectsNode(host)) {
        return null
      }
    } catch {
      return null
    }

    const wrapped = document.createRange()

    wrapped.selectNodeContents(host)

    return wrapped.collapsed ? null : wrapped
  }

  const hostRange = document.createRange()

  hostRange.selectNodeContents(host)

  const clipped = range.cloneRange()

  if (!startIn) {
    clipped.setStart(hostRange.startContainer, hostRange.startOffset)
  }

  if (!endIn) {
    clipped.setEnd(hostRange.endContainer, hostRange.endOffset)
  }

  return clipped.collapsed ? null : clipped
}

/** Live window selection that intersects `host` (collapsed / outside → null). */
export function readHostTextSelection(host: HTMLElement): { range: Range; text: string } | null {
  const selection = window.getSelection()

  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return null
  }

  const clipped = clipRangeToHost(host, selection.getRangeAt(0))

  if (!clipped) {
    return null
  }

  const text = clipped.toString()

  if (!text.trim()) {
    return null
  }

  return { range: clipped, text }
}

/** True when the live window selection intersects a preview Add-to-Chat frame. */
export function selectionBelongsToPreviewAddToChat(): boolean {
  const selection = window.getSelection()

  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return false
  }

  for (const frame of document.querySelectorAll<HTMLElement>(`[${PREVIEW_ADD_TO_CHAT_ATTR}]`)) {
    if (readHostTextSelection(frame)) {
      return true
    }
  }

  return false
}

/**
 * Resolve a DOM range to source lines. Prefers exact selected-text match; when
 * the rendered DOM diverges (virtualized chunks, highlighting, drag past the
 * frame), falls back to gutter / prefix offsets from the range geometry.
 */
export function lineSelectionFromHostRange(
  fullText: string,
  host: HTMLElement,
  range: Range,
  selectedText?: string
): LineSelection | null {
  const text = selectedText ?? range.toString()
  const preferOffset = preferOffsetFromRange(fullText, host, range)
  const fromText = lineSelectionFromSelectedText(fullText, text, preferOffset)

  if (fromText) {
    return fromText
  }

  const startRange = range.cloneRange()

  startRange.collapse(true)

  const endRange = range.cloneRange()

  endRange.collapse(false)

  const startOffset = preferOffsetFromRange(fullText, host, startRange) ?? preferOffset
  const endOffset = preferOffsetFromRange(fullText, host, endRange) ?? preferOffset

  if (startOffset == null && endOffset == null) {
    return null
  }

  const start = startOffset ?? endOffset ?? 0
  const end = Math.max((endOffset ?? start) + 1, start + 1)

  return lineSelectionFromOffsets(fullText, start, end)
}

/** Char offset of the start of a 1-based line in newline-normalized text. */
export function offsetOfLineStart(fullText: string, line1Based: number): number {
  const text = normalizeNewlines(fullText)

  if (line1Based <= 1) {
    return 0
  }

  let line = 1

  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n') {
      line += 1

      if (line === line1Based) {
        return i + 1
      }
    }
  }

  return text.length
}

/**
 * Prefer-offset for duplicate-needle disambiguation. Uses nearby
 * `[data-preview-line]` gutter markers when present (virtualized source);
 * otherwise falls back to the DOM prefix length inside `host`.
 */
export function preferOffsetFromRange(source: string, host: HTMLElement, range: Range): number | undefined {
  const text = normalizeNewlines(source)

  if (!text) {
    return undefined
  }

  try {
    const rect = range.getBoundingClientRect()
    const y = rect.height > 0 ? rect.top + Math.min(4, rect.height / 2) : rect.top
    const markers = host.querySelectorAll<HTMLElement>('[data-preview-line]')
    let bestLine: number | null = null
    let bestDist = Infinity

    for (const el of markers) {
      const markerRect = el.getBoundingClientRect()

      if (markerRect.height <= 0) {
        continue
      }

      const mid = (markerRect.top + markerRect.bottom) / 2
      const dist = Math.abs(mid - y)

      if (dist < bestDist) {
        bestDist = dist
        const line = Number(el.dataset.previewLine)

        if (Number.isFinite(line) && line >= 1) {
          bestLine = line
        }
      }
    }

    if (bestLine != null) {
      return offsetOfLineStart(text, bestLine)
    }

    const pre = document.createRange()

    pre.selectNodeContents(host)
    pre.setEnd(range.startContainer, range.startOffset)

    return normalizeNewlines(pre.toString()).length
  } catch {
    return undefined
  }
}

/**
 * When the preview frame has a claimable line/text selection, it owns ⌘/Ctrl+L.
 * The terminal's long-lived capture listener registers earlier and would otherwise
 * also insert an `@terminal:` quote from the same `window.getSelection()`.
 */
let previewAddShortcutClaims = 0

export function retainPreviewAddShortcutClaim(): () => void {
  previewAddShortcutClaims += 1

  return () => {
    previewAddShortcutClaims = Math.max(0, previewAddShortcutClaims - 1)
  }
}

export function previewOwnsAddSelectionShortcut(): boolean {
  return previewAddShortcutClaims > 0 || selectionBelongsToPreviewAddToChat()
}
