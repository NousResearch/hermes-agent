// List continuation for Shift+Enter in the composer.
//
// The composer is a plain contentEditable with no structured list model, so
// list items are just lines of text starting with a marker (`1.`, `1)`, `-`,
// `*`, `+`). When the user presses Shift+Enter on a list line, the native
// newline is replaced by a newline plus the NEXT marker, the way Notion/Obsidian
// continue a list. This file holds the marker math (pure, unit-tested) and one
// DOM commit path built on the same caret helpers the URL-chip conversion uses.

import { revealCaret, serializeTextBefore } from './rich-editor'
import { markdownCodeRanges } from './url-refs'

export type ListLineKind = 'numbered' | 'bulleted'

/** The list line ending at the caret, with marker components. */
export interface ListLineMatch {
  indent: string
  /** The marker as typed, trailing space included: `1. ` / `- `. */
  marker: string
  kind: ListLineKind
  /** `.` or `)` for numbered lines; '' for bulleted. */
  delimiter: string
  /** The number for a numbered line; null for bulleted. */
  number: number | null
  /** Everything after the marker on the line (may be ''). */
  rest: string
}

const LINE_MARKER_RE = /^(\s*)(\d{1,9}[.)]|[*+-])(?:[ \t]+(.*))?$/

/** Parse the list line ending at the caret, or null when the caret's line is
 *  not a list item. A line that is only indent + marker (`1.` + Enter) parses
 *  too, with `rest` empty — the "end the list" case. A marker with no
 *  following space (`12abc`, `-text`) is not a marker, just prose. */
export function parseListLineBefore(before: string): ListLineMatch | null {
  const lineStart = before.lastIndexOf('\n') + 1
  const line = before.slice(lineStart)

  const match = LINE_MARKER_RE.exec(line)

  if (!match) {
    return null
  }

  const indent = match[1]
  const marker = match[2]
  const rest = match[3] ?? ''
  const numbered = /^(\d{1,9})([.)])$/.exec(marker)

  if (numbered) {
    return {
      indent,
      marker,
      kind: 'numbered',
      delimiter: numbered[2],
      number: Number.parseInt(numbered[1], 10),
      rest
    }
  }

  return { indent, marker, kind: 'bulleted', delimiter: '', number: null, rest }
}

/** The marker the next line starts with: bulleted keeps its bullet; numbered
 *  increments by 1 (long `10.` runs are not a real composer use case). */
export function nextListMarker(match: ListLineMatch): string {
  if (match.kind === 'bulleted') {
    return match.marker + ' '
  }

  return `${(match.number ?? 0) + 1}${match.delimiter} `
}

/** When continuing a list, we don't delete anything - we just insert a newline
 *  + the next marker after the current line. The marker-only case is the same:
 *  insert a newline, which ends the list (no marker on the next line). */
export function continuationDeleteLength(_match: ListLineMatch): number {
  return 0
}

/** Commit the continuation for a collapsed caret: insert `<br>` + indent +
 *  next marker after the current line (or just `<br>` when the line is
 *  marker-only, ending the list). Leaves the caret after the insertion.
 *  Returns whether it ran (callers preventDefault on true). */
export function continueListAtCaret(editor: HTMLElement): boolean {
  const selection = window.getSelection()
  const range = selection?.rangeCount ? selection.getRangeAt(0) : null

  if (!range || !range.collapsed || !editor.contains(range.commonAncestorContainer)) {
    return false
  }

  const before = serializeTextBefore(editor, range.startContainer, range.startOffset)
  const match = parseListLineBefore(before)

  if (!match) {
    return false
  }

  // A marker typed inside inline code / a fenced block is code, not a list.
  const lineStart = before.lastIndexOf('\n') + 1
  const inCode = markdownCodeRanges(before).some(r => lineStart >= r.start && lineStart < r.end)

  if (inCode) {
    return false
  }

  const br = document.createElement('br')
  const fragment = document.createDocumentFragment()

  fragment.append(br)

  // When the line has content after the marker, continue the list; when it's
  // marker-only, the newline ends the list (no marker on the next line).
  const next = match.rest ? `${match.indent}${nextListMarker(match)}` : ''

  let textNode: Text | null = null

  if (next) {
    textNode = document.createTextNode(next)
    fragment.append(textNode)
  }

  // Insert at the caret position (delete nothing). insertNode moves the
  // fragment's children into the DOM, so capture the text node reference
  // BEFORE the insertion to position the caret after it.
  const insertRange = document.createRange()
  insertRange.setStart(range.startContainer, range.startOffset)
  insertRange.setEnd(range.startContainer, range.startOffset)
  insertRange.insertNode(fragment)

  // Move the caret to the end of the inserted content
  const anchor = textNode ?? br

  if (anchor && editor.contains(anchor)) {
    const newCaretRange = document.createRange()

    if (textNode) {
      // Caret after the text node (after the marker)
      newCaretRange.setStart(textNode, textNode.textContent?.length ?? 0)
    } else {
      // Marker-only case: caret after the <br>
      newCaretRange.setStartAfter(anchor)
    }

    newCaretRange.collapse(true)
    selection?.removeAllRanges()
    selection?.addRange(newCaretRange)
  }

  revealCaret(editor)

  return true
}
