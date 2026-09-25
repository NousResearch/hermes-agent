/**
 * Markdown list editing for the composer: what Shift+Enter, Tab, Shift+Tab and
 * Backspace do when the caret is on a list line. Pure text in, text out, so the
 * rules are testable without a DOM; `app/chat/composer/list-keys.ts` applies
 * them to the editor.
 *
 * Numbers (`1.` `1)`), bullets (`-` `*` `+`) and task boxes are CommonMark/GFM
 * and always on. The extra styles are not Markdown lists (they reach the model
 * as plain text), so each is opt-in from Settings → Appearance.
 *
 * Nesting follows CommonMark: a child is indented to its parent's content
 * column (`1. ` → 3 spaces, `- ` → 2), or it would not render as a child. A new
 * child restarts its sequence in the parent's style (`2.` → `1.`, `b.` → `a.`).
 */

export const EXTRA_LIST_STYLES = ['letters', 'roman', 'outline', 'parentheses', 'symbols'] as const

export type ExtraListStyle = (typeof EXTRA_LIST_STYLES)[number]

/** Shift+Enter, Backspace, Tab, Shift+Tab. Plain Enter always sends and never gets here. */
export type ListKey = 'backspace' | 'indent' | 'newline' | 'outdent'

export interface ListEdit {
  caret: number
  text: string
}

type Kind = 'bullet' | 'letter' | 'number' | 'outline' | 'roman'

interface ListItem {
  box: string
  bullet: string
  close: string
  /** Where the item's text starts, after the marker, its spaces and any task box. */
  contentStart: number
  indent: string
  /** Resolved style; `i`, `v`, `x` are both letters and numerals (see `resolveKind`). */
  kinds: Kind[]
  /** Width of the marker plus its spaces: the indent a child needs to nest. */
  markerWidth: number
  open: string
  rest: string
  token: string
}

interface Line {
  end: number
  start: number
  text: string
}

// indent, then `(`token`)` / token`.` / token`)` / a bullet, 1+ spaces, an optional task box.
const ITEM_RE = /^([ \t]*)(?:(\()?(\d{1,9}(?:\.\d{1,9})*|[A-Za-z]{1,7})([.)])|([-*+•◦▪‣]))( +)(\[[ xX]\] +)?(.*)$/
const MARKDOWN_BULLETS = '-*+'
const FENCE_RE = /^ {0,3}(`{3,}|~{3,})/

const ROMAN: readonly [number, string][] = [
  [90, 'xc'],
  [50, 'l'],
  [40, 'xl'],
  [10, 'x'],
  [9, 'ix'],
  [5, 'v'],
  [4, 'iv'],
  [1, 'i']
]

const isUpper = (token: string) => token === token.toUpperCase()

/** `a`..`z` then `aa`..`zz` (the Word / Google Docs sequence); 0 for anything else. */
function letterIndex(token: string): number {
  const match = /^([a-z])\1?$/.exec(token) ?? /^([A-Z])\1?$/.exec(token)

  return match ? (token.length - 1) * 26 + token.toLowerCase().charCodeAt(0) - 96 : 0
}

function letterAt(index: number, upper: boolean): null | string {
  if (index < 1 || index > 52) {
    return null
  }

  const letter = String.fromCharCode(97 + ((index - 1) % 26)).repeat(index > 26 ? 2 : 1)

  return upper ? letter.toUpperCase() : letter
}

function toRoman(value: number): string {
  let rest = value
  let numeral = ''

  for (const [step, glyph] of ROMAN) {
    while (rest >= step) {
      numeral += glyph
      rest -= step
    }
  }

  return numeral
}

/** Canonical numeral in one case, 1..99; 0 otherwise. The cap keeps `mix.` or `MD.` from reading as numbers. */
function romanValue(token: string): number {
  if (!/^[ivxl]+$/.test(token) && !/^[IVXL]+$/.test(token)) {
    return 0
  }

  for (let value = 1; value < 100; value++) {
    if (toRoman(value) === token.toLowerCase()) {
      return value
    }
  }

  return 0
}

function parseItem(line: string, styles: ReadonlySet<ExtraListStyle>): ListItem | null {
  const match = ITEM_RE.exec(line)

  if (!match) {
    return null
  }

  const [, indent = '', open = '', token = '', close = '', bullet = '', spaces = '', box = '', rest = ''] = match

  if (open && (close !== ')' || !styles.has('parentheses'))) {
    return null
  }

  let kinds: Kind[]

  if (bullet) {
    kinds = MARKDOWN_BULLETS.includes(bullet) || styles.has('symbols') ? ['bullet'] : []
  } else if (/^\d+$/.test(token)) {
    kinds = ['number']
  } else if (/^\d/.test(token)) {
    kinds = styles.has('outline') ? ['outline'] : []
  } else {
    kinds = [
      ...(styles.has('letters') && letterIndex(token) ? (['letter'] as const) : []),
      ...(styles.has('roman') && romanValue(token) ? (['roman'] as const) : [])
    ]
  }

  if (!kinds.length) {
    return null
  }

  const markerWidth = open.length + (bullet || token).length + close.length + spaces.length

  return {
    box,
    bullet,
    close,
    contentStart: indent.length + markerWidth + box.length,
    indent,
    kinds,
    markerWidth,
    open,
    rest,
    token
  }
}

/** One style for an item: `h.` → `i.` stays letters and `iv.` → `v.` stays numerals, decided by the
 *  item above; a list that starts at `i` is Roman, any other single letter is a letter list. */
function resolveKind(item: ListItem, above: ListItem | null): Kind {
  if (item.kinds.length === 1) {
    return item.kinds[0]!
  }

  if (above && above.kinds.length && isUpper(above.token) === isUpper(item.token)) {
    if (romanValue(above.token) && romanValue(above.token) === romanValue(item.token) - 1) {
      return 'roman'
    }

    if (letterIndex(above.token) === letterIndex(item.token) - 1) {
      return 'letter'
    }
  }

  return /^i$/i.test(item.token) || item.token.length > 1 ? 'roman' : 'letter'
}

function nextToken(item: ListItem, kind: Kind): null | string {
  const { token } = item

  switch (kind) {
    case 'bullet':
      return item.bullet

    case 'number':
      return String(Number(token) + 1).padStart(token.length, '0')
    case 'outline': {
      const parts = token.split('.')
      parts[parts.length - 1] = String(Number(parts.at(-1)) + 1)

      return parts.join('.')
    }

    case 'letter':
      return letterAt(letterIndex(token) + 1, isUpper(token))
    case 'roman': {
      const value = romanValue(token) + 1

      return value < 100 ? (isUpper(token) ? toRoman(value).toUpperCase() : toRoman(value)) : null
    }
  }
}

/** The first marker of a new list level; an outline child extends its parent's number (`1.2.` → `1.2.1.`). */
function firstToken(item: ListItem, kind: Kind, parent: ListItem | null): string {
  switch (kind) {
    case 'bullet':
      return item.bullet

    case 'number':
      return '1'

    case 'outline':
      return parent && /^\d/.test(parent.token) ? `${parent.token}.1` : item.token.replace(/\.\d+$/, '')

    case 'letter':
      return isUpper(item.token) ? 'A' : 'a'

    case 'roman':
      return isUpper(item.token) ? 'I' : 'i'
  }
}

const markerText = (item: ListItem, token: string, box: string) => `${item.open}${token}${item.close} ${box}`

function linesOf(text: string): Line[] {
  const lines: Line[] = []
  let start = 0

  for (const part of text.split('\n')) {
    lines.push({ end: start + part.length, start, text: part })
    start += part.length + 1
  }

  return lines
}

function insideCodeFence(lines: readonly Line[], index: number): boolean {
  let open = false

  for (let i = 0; i < index; i++) {
    if (FENCE_RE.test(lines[i]!.text)) {
      open = !open
    }
  }

  return open
}

/** Line index of the nearest item above `index` at exactly `indent` columns, in the same list; -1 if none. */
function itemIndexAt(
  lines: readonly Line[],
  index: number,
  indent: number,
  styles: ReadonlySet<ExtraListStyle>
): number {
  for (let i = index - 1; i >= 0; i--) {
    const text = lines[i]!.text

    if (!text.trim()) {
      continue
    }

    const item = parseItem(text, styles)
    const depth = item ? item.indent.length : text.length - text.trimStart().length

    if (depth > indent) {
      continue
    }

    return depth === indent && item ? i : -1
  }

  return -1
}

function itemAt(lines: readonly Line[], index: number, indent: number, styles: ReadonlySet<ExtraListStyle>) {
  const at = itemIndexAt(lines, index, indent, styles)

  return at < 0 ? null : parseItem(lines[at]!.text, styles)
}

/** The style of the item on line `index`, with the item above it as the tie-breaker. */
function kindAt(lines: readonly Line[], index: number, item: ListItem, styles: ReadonlySet<ExtraListStyle>): Kind {
  return resolveKind(item, itemAt(lines, index, item.indent.length, styles))
}

/** The item a line at `indent` nests under: the nearest item above with a smaller indent. */
function parentOf(
  lines: readonly Line[],
  index: number,
  indent: number,
  styles: ReadonlySet<ExtraListStyle>
): ListItem | null {
  for (let i = index - 1; i >= 0; i--) {
    const text = lines[i]!.text

    if (!text.trim()) {
      continue
    }

    const item = parseItem(text, styles)

    if (item && item.indent.length < indent) {
      return item
    }

    if (!item && text.length - text.trimStart().length < indent) {
      return null
    }
  }

  return null
}

function replaceLine(text: string, line: Line, next: string, caret: number): ListEdit {
  return { caret, text: text.slice(0, line.start) + next + text.slice(line.end) }
}

/** Re-home `item` at `indent`: continue the list already at that level, or start one there. */
function moveItem(
  text: string,
  lines: readonly Line[],
  index: number,
  item: ListItem,
  indent: string,
  caret: number,
  styles: ReadonlySet<ExtraListStyle>
): ListEdit {
  const line = lines[index]!
  const parent = indent ? parentOf(lines, index, indent.length, styles) : null
  const siblingIndex = itemIndexAt(lines, index, indent.length, styles)
  const sibling = siblingIndex < 0 ? null : parseItem(lines[siblingIndex]!.text, styles)
  const token = sibling ? nextToken(sibling, kindAt(lines, siblingIndex, sibling, styles)) : null

  // The item takes the style of the list it lands in; a new level restarts in its own style.
  const marker =
    sibling && token !== null
      ? markerText(sibling, token, item.box)
      : markerText(item, firstToken(item, kindAt(lines, index, item, styles), parent), item.box)

  const next = `${indent}${marker}${item.rest}`
  const offset = Math.max(0, caret - line.start - item.contentStart)

  return replaceLine(text, line, next, line.start + indent.length + marker.length + offset)
}

/**
 * The edit a list key makes at `caret` in `text`, or null when the key is not a
 * list edit there (not a list line, inside a code fence, caret in the marker)
 * and should keep its normal meaning. A key on a list line that has nothing to
 * do (Tab on a first item, Shift+Tab at the top level) returns the text
 * unchanged, so Tab never moves focus out of a list.
 */
export function listEdit(
  text: string,
  caret: number,
  key: ListKey,
  styles: ReadonlySet<ExtraListStyle> = new Set()
): ListEdit | null {
  const lines = linesOf(text)
  const index = lines.findIndex(line => caret >= line.start && caret <= line.end)
  const line = lines[index]

  if (!line || insideCodeFence(lines, index)) {
    return null
  }

  const item = parseItem(line.text, styles)

  if (!item) {
    return null
  }

  const column = caret - line.start
  const blank = !item.rest.trim()

  if (key === 'indent') {
    const parent = itemAt(lines, index, item.indent.length, styles)

    return parent
      ? moveItem(text, lines, index, item, parent.indent + ' '.repeat(parent.markerWidth), caret, styles)
      : { caret, text }
  }

  if (key === 'outdent') {
    const parent = item.indent ? parentOf(lines, index, item.indent.length, styles) : null

    return item.indent ? moveItem(text, lines, index, item, parent?.indent ?? '', caret, styles) : { caret, text }
  }

  if (key === 'backspace') {
    if (!blank || column !== line.text.length) {
      return null
    }

    // The indent stays: under a parent it now holds a paragraph of that item.
    return replaceLine(text, line, item.indent, line.start + item.indent.length)
  }

  if (column < item.contentStart) {
    return null
  }

  if (blank) {
    if (item.indent) {
      const parent = parentOf(lines, index, item.indent.length, styles)

      return moveItem(text, lines, index, item, parent?.indent ?? '', caret, styles)
    }

    // Leaving the list at the top level keeps one blank line: without it,
    // Markdown reads the next line as more text of the last item.
    const last = index === lines.length - 1

    return {
      caret: line.start + (last ? 1 : 0),
      text: text.slice(0, line.start) + (last ? '\n' : '') + text.slice(line.end)
    }
  }

  const token = nextToken(item, kindAt(lines, index, item, styles))

  if (token === null) {
    return null
  }

  // The item's text never starts with a space (the marker regex takes them all),
  // so trimming what stays behind can't reach into the marker.
  const head = column > item.contentStart ? line.text.slice(0, column).trimEnd() : line.text.slice(0, column)
  const marker = `${item.indent}${markerText(item, token, item.box ? '[ ] ' : '')}`
  const tail = line.text.slice(column).trimStart()

  return {
    caret: line.start + head.length + 1 + marker.length,
    text: `${text.slice(0, line.start)}${head}\n${marker}${tail}${text.slice(line.end)}`
  }
}
