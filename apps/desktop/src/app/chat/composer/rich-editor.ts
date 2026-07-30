/**
 * Helpers for the contenteditable composer surface: serialize refs to chip
 * HTML, walk the DOM back to plain `@kind:value` text, and place the caret.
 *
 * Chip values are always wrapped in backticks/quotes so REF_RE stops at the
 * fence — without that, typing after a chip would get re-absorbed on the next
 * plain-text round-trip.
 */
import {
  DIRECTIVE_CHIP_CLASS,
  directiveIconElement,
  directiveIconSvg,
  formatRefValue,
  refChipLabel,
  slashChipClass,
  type SlashChipKind,
  slashIconElement
} from '@/components/assistant-ui/directive-text'
import {
  desktopSlashCommandArgumentMode,
  isDesktopSlashCommand,
  resolveDesktopCommand
} from '@/lib/desktop-slash-commands'

export const RICH_INPUT_SLOT = 'composer-rich-input'

export const REF_RE = /@(file|folder|url|image|tool|line|terminal|session):(`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)/g

/** A committed leading slash command: `/name` followed by whitespace. The
 *  whitespace requirement is what separates a committed command (chips always
 *  serialize with their auto-inserted trailing space) from one still being
 *  typed, which must stay editable text. */
const LEADING_SLASH_COMMAND_RE = /^\/[a-zA-Z][\w-]*(?=\s)/

const ESC: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }

export function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, ch => ESC[ch] || ch)
}

export function unquoteRef(raw: string) {
  const head = raw[0]
  const tail = raw[raw.length - 1]
  const quoted = (head === '`' && tail === '`') || (head === '"' && tail === '"') || (head === "'" && tail === "'")

  return quoted ? raw.slice(1, -1) : raw.replace(/[,.;!?]+$/, '')
}

/** Always-quote variant of formatRefValue — chips need a fence even for safe values. */
export function quoteRefValue(value: string) {
  if (!value.includes('`')) {
    return `\`${value}\``
  }

  if (!value.includes('"')) {
    return `"${value}"`
  }

  if (!value.includes("'")) {
    return `'${value}'`
  }

  return formatRefValue(value)
}

export function refChipHtml(kind: string, rawValue: string, displayLabel?: string) {
  const id = unquoteRef(rawValue)
  const text = `@${kind}:${quoteRefValue(id)}`

  const label = displayLabel || refChipLabel(kind, id)

  return `<span contenteditable="false" title="${escapeHtml(id)}" data-ref-text="${escapeHtml(text)}" data-ref-id="${escapeHtml(id)}" data-ref-kind="${escapeHtml(kind)}" class="${DIRECTIVE_CHIP_CLASS}">${directiveIconSvg(kind)}<span class="truncate">${escapeHtml(label)}</span></span>`
}

export function refChipElement(kind: string, rawValue: string, displayLabel?: string) {
  const id = unquoteRef(rawValue)
  const text = `@${kind}:${quoteRefValue(id)}`
  const chip = document.createElement('span')
  const label = document.createElement('span')

  chip.contentEditable = 'false'
  chip.title = id
  chip.dataset.refText = text
  chip.dataset.refId = id
  chip.dataset.refKind = kind
  chip.className = DIRECTIVE_CHIP_CLASS
  label.className = 'truncate'
  label.textContent = displayLabel || refChipLabel(kind, id)
  chip.append(directiveIconElement(kind), label)

  return chip
}

/** A non-editable pill for a picked slash command (`/skin nous`, `/tropes`).
 *  `data-ref-text` carries the literal command so `composerPlainText` round-trips
 *  it back to the exact text that gets submitted. */
export function slashChipElement(command: string, kind: SlashChipKind, label?: string) {
  const chip = document.createElement('span')
  const text = document.createElement('span')

  chip.contentEditable = 'false'
  chip.dataset.refText = command
  chip.dataset.slashKind = kind
  chip.className = slashChipClass(kind)
  text.className = 'truncate'
  text.textContent = label || command
  chip.append(slashIconElement(kind), text)

  return chip
}

function appendTextWithBreaks(target: DocumentFragment | HTMLElement, text: string) {
  const lines = text.split('\n')

  lines.forEach((line, index) => {
    if (index > 0) {
      target.append(document.createElement('br'))
    }

    if (line) {
      target.append(document.createTextNode(line))
    }
  })
}

export function appendComposerContents(target: DocumentFragment | HTMLElement, text: string) {
  let cursor = 0

  REF_RE.lastIndex = 0

  for (const match of text.matchAll(REF_RE)) {
    const index = match.index ?? 0
    appendTextWithBreaks(target, text.slice(cursor, index))
    target.append(refChipElement(match[1] || 'file', match[2] || ''))
    cursor = index + match[0].length
  }

  appendTextWithBreaks(target, text.slice(cursor))
}

export function renderComposerContents(target: HTMLElement, text: string) {
  target.replaceChildren()

  // A leading `/command` hydrates back to its pill — parity with REF_RE for
  // `@` refs, so a full re-render from serialized text (draft restore, undo,
  // the trigger commit fallback) doesn't demote a committed command chip to
  // plain text. Only commands with NO argument stage qualify (skills, quick
  // commands, no-arg built-ins): their committed pill is exactly the bare
  // `/name`, so the boundary is unambiguous. Arg-taking commands (`/goal ship
  // it`, `/personality alice`) stay text — their tail may be prose that was
  // never committed. The trailing whitespace is load-bearing too: a committed
  // pill always serializes with its auto-inserted space, while a half-typed
  // `/wor` must stay editable text.
  const command = LEADING_SLASH_COMMAND_RE.exec(text)?.[0]

  if (command && isDesktopSlashCommand(command) && desktopSlashCommandArgumentMode(command) === null) {
    target.append(slashChipElement(command, resolveDesktopCommand(command) ? 'command' : 'skill'))
    text = text.slice(command.length)
  }

  appendComposerContents(target, text)
}

/** Caret range when the selection lives inside `editor`; else null. */
function composerSelectionRange(editor: HTMLElement) {
  const selection = window.getSelection()
  const range = selection?.rangeCount ? selection.getRangeAt(0) : null

  if (!selection || !range || !editor.contains(range.commonAncestorContainer)) {
    return null
  }

  return { range, selection }
}

/** Insert text at the caret (replacing any selection), with any `@kind:value`
 *  directives in it landing as chips. Pastes use this instead of
 *  `execCommand('insertText')` — Chromium's editing pipeline is ~O(n²) on large
 *  multiline blobs. */
export function insertComposerContentsAtCaret(editor: HTMLElement, text: string) {
  const hit = composerSelectionRange(editor)
  const fragment = document.createDocumentFragment()

  appendComposerContents(fragment, text)

  const tail = fragment.lastChild

  if (hit) {
    hit.range.deleteContents()
    hit.range.insertNode(fragment)
  } else {
    editor.append(fragment)
  }

  if (tail) {
    const caret = document.createRange()
    caret.setStartAfter(tail)
    caret.collapse(true)
    const selection = hit?.selection ?? window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(caret)
  }
}

/** Range covering exactly `length` serialized characters immediately before a
 *  collapsed caret, spanning Chromium's split text nodes. Null when the caret
 *  isn't a collapsed selection in `editor`, or when a chip/<br>/block boundary
 *  interrupts before `length` characters are covered — a trigger token is
 *  always contiguous text, so anything else means "don't touch the DOM here".
 *
 *  This is what keeps chip insertion stable: Chromium fragments text nodes
 *  around contenteditable=false chips on every edit, so any commit path that
 *  demands the whole token inside ONE text node (the old check) degrades to a
 *  full re-render as soon as a chip exists anywhere in the line. */
export function rangeBeforeCaret(editor: HTMLElement, length: number): Range | null {
  const hit = composerSelectionRange(editor)

  if (!hit?.range.collapsed || length <= 0) {
    return null
  }

  let node: Node | null = hit.range.startContainer
  let offset = hit.range.startOffset

  // An element-positioned caret (common right after programmatic caret moves)
  // resolves to the end of the text node before it. A chip or <br> there means
  // no text token precedes the caret — bail rather than guess.
  if (node.nodeType !== Node.TEXT_NODE) {
    node = node.childNodes[offset - 1] ?? null

    if (node?.nodeType !== Node.TEXT_NODE) {
      return null
    }

    offset = (node.textContent || '').length
  }

  let startNode = node as Text
  let startOffset = offset
  let remaining = length

  while (remaining > 0) {
    if (startOffset >= remaining) {
      startOffset -= remaining
      remaining = 0

      break
    }

    remaining -= startOffset

    const prev: Node | null = startNode.previousSibling

    if (prev?.nodeType !== Node.TEXT_NODE) {
      return null
    }

    startNode = prev as Text
    startOffset = (prev.textContent || '').length
  }

  const range = document.createRange()

  range.setStart(startNode, startOffset)
  range.setEnd(hit.range.startContainer, hit.range.startOffset)

  return range
}

/** Swap the `length` characters immediately before a collapsed caret for
 *  `fragment`, leaving the caret after it. Returns whether it ran. Spans split
 *  text nodes (see rangeBeforeCaret) — a token typed around existing chips
 *  still commits in place instead of falling back to a full re-render. */
export function replaceBeforeCaret(editor: HTMLElement, length: number, fragment: DocumentFragment) {
  const range = rangeBeforeCaret(editor, length)

  if (!range) {
    return false
  }

  const tail = fragment.lastChild

  range.deleteContents()
  range.insertNode(fragment)

  if (tail) {
    range.setStartAfter(tail)
  }

  range.collapse(true)

  const selection = window.getSelection()

  selection?.removeAllRanges()
  selection?.addRange(range)

  return true
}

/** Backspace at a collapsed caret immediately after a chip: delete the chip AND
 *  the single trailing space we auto-insert after it, atomically — so removing a
 *  directive never strands an orphaned space (the contenteditable-driven cleanup
 *  was unreliable). Returns whether it ran. */
export function deleteChipBeforeCaret(editor: HTMLElement): boolean {
  const hit = composerSelectionRange(editor)

  if (!hit || !hit.range.collapsed) {
    return false
  }

  const { startContainer, startOffset } = hit.range
  let chip: ChildNode | null = null

  if (startContainer === editor) {
    chip = startOffset > 0 ? editor.childNodes[startOffset - 1] : null
  } else if (startContainer.nodeType === Node.TEXT_NODE && startOffset === 0) {
    chip = startContainer.previousSibling
  }

  if (chip?.nodeType !== Node.ELEMENT_NODE || !(chip as HTMLElement).dataset.refText) {
    return false
  }

  const after = chip.nextSibling
  chip.remove()

  // Drop the auto-inserted trailing space; keep any real following text.
  if (after?.nodeType === Node.TEXT_NODE) {
    const text = after.textContent ?? ''

    if (text === ' ') {
      after.remove()
    } else if (text.startsWith(' ')) {
      after.textContent = text.slice(1)
    }
  }

  const caret = document.createRange()

  if (after?.isConnected) {
    caret.setStartBefore(after)
  } else {
    caret.selectNodeContents(editor)
    caret.collapse(false)
  }

  caret.collapse(true)
  hit.selection.removeAllRanges()
  hit.selection.addRange(caret)

  return true
}

/** Remove a non-collapsed selection in-editor. Skips collapsed carets so word/
 *  line delete (Opt/Cmd+Backspace) stays native. Returns whether anything ran. */
export function deleteSelectionInEditor(editor: HTMLElement) {
  const hit = composerSelectionRange(editor)

  if (!hit || hit.range.collapsed) {
    return false
  }

  hit.range.deleteContents()
  hit.range.collapse(true)
  hit.selection.removeAllRanges()
  hit.selection.addRange(hit.range)

  return true
}

/** Serialize a draft string into chip-HTML for the contenteditable surface. */
export function composerHtml(text: string) {
  let cursor = 0
  let html = ''

  REF_RE.lastIndex = 0

  for (const match of text.matchAll(REF_RE)) {
    const index = match.index ?? 0
    html += escapeHtml(text.slice(cursor, index)).replace(/\n/g, '<br>')
    html += refChipHtml(match[1] || 'file', match[2] || '')
    cursor = index + match[0].length
  }

  return html + escapeHtml(text.slice(cursor)).replace(/\n/g, '<br>')
}

/** Walk a DOM subtree back to the plain `@kind:value` text it represents. */
export function composerPlainText(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) {
    return node.textContent || ''
  }

  if (node.nodeType !== Node.ELEMENT_NODE) {
    return ''
  }

  const el = node as HTMLElement

  if (el.dataset.refText) {
    return el.dataset.refText
  }

  if (el.tagName === 'BR') {
    return '\n'
  }

  const text = Array.from(node.childNodes).map(composerPlainText).join('')
  const block = el.tagName === 'DIV' || el.tagName === 'P'

  return block && text && el.dataset.slot !== RICH_INPUT_SLOT ? `${text}\n` : text
}

export function placeCaretEnd(element: HTMLElement) {
  const range = document.createRange()
  const selection = window.getSelection()

  range.selectNodeContents(element)
  range.collapse(false)
  selection?.removeAllRanges()
  selection?.addRange(range)
}

/** The caret's offset in `composerPlainText` coordinates, so it can be restored
 *  after the editor is re-rendered from text (undo/redo). A chip counts as its
 *  whole `@kind:value` text — the same units the snapshot measures. */
export function caretOffsetInEditor(editor: HTMLElement): number {
  const selection = window.getSelection()
  const range = selection?.rangeCount ? selection.getRangeAt(0) : null

  if (!range || !editor.contains(range.commonAncestorContainer)) {
    return composerPlainText(editor).length
  }

  const before = range.cloneRange()
  before.selectNodeContents(editor)
  before.setEnd(range.startContainer, range.startOffset)

  // The scratch container must carry the editor's slot marker: composerPlainText
  // appends a trailing "\n" to any other block element, which would inflate
  // every offset by one and land the restored caret a character late.
  const container = document.createElement('div')
  container.dataset.slot = RICH_INPUT_SLOT
  container.append(before.cloneContents())

  return composerPlainText(container).length
}

/** Place the caret `offset` characters into the editor, in the same
 *  `composerPlainText` coordinates `caretOffsetInEditor` reports. Lands after a
 *  chip it would otherwise split, since a chip is a single atomic unit. */
export function placeCaretAtOffset(editor: HTMLElement, offset: number) {
  const selection = window.getSelection()

  if (!selection) {
    return
  }

  let remaining = offset

  const walk = (node: Node): Range | null => {
    for (const child of Array.from(node.childNodes)) {
      if (child.nodeType === Node.TEXT_NODE) {
        const length = (child.textContent || '').length

        if (remaining <= length) {
          const range = document.createRange()
          range.setStart(child, remaining)
          range.collapse(true)

          return range
        }

        remaining -= length

        continue
      }

      if (child.nodeType !== Node.ELEMENT_NODE) {
        continue
      }

      const el = child as HTMLElement

      // Chips and <br> are atomic: consume their serialized length whole.
      if (el.dataset.refText || el.tagName === 'BR') {
        const length = el.dataset.refText ? el.dataset.refText.length : 1

        if (remaining < length) {
          const range = document.createRange()
          range.setStartBefore(el)
          range.collapse(true)

          return range
        }

        remaining -= length

        continue
      }

      const hit = walk(el)

      if (hit) {
        return hit
      }
    }

    return null
  }

  const range = walk(editor)

  if (range) {
    selection.removeAllRanges()
    selection.addRange(range)

    return
  }

  placeCaretEnd(editor)
}

/** Nothing but a break / whitespace (recursively) — i.e. no real text or chip. */
function isBlankNode(node: ChildNode | null): boolean {
  if (!node) {
    return false
  }

  if (node.nodeName === 'BR') {
    return true
  }

  if (node.nodeType === Node.TEXT_NODE) {
    return !(node.textContent || '').trim()
  }

  if (node.nodeType === Node.ELEMENT_NODE) {
    const el = node as HTMLElement

    return !el.dataset.refText && Array.from(el.childNodes).every(isBlankNode)
  }

  return false
}

/** Drop contenteditable junk that serializes as `\n` and falsely expands the
 *  composer. Editing around a contenteditable=false chip makes Chromium wrap the
 *  remainder in stray block <div>s / trailing <br>s — none of which our own
 *  rendering emits (we use text nodes + <br> + chips). Real <br> line breaks
 *  (Shift+Enter, which sit after actual text) are preserved. */
export function normalizeComposerEditorDom(editor: HTMLElement) {
  // A trailing block wrapper holding only a break/whitespace is the phantom
  // "new line" Chromium adds after a chip on backspace — drop it.
  const tailBlock = editor.lastChild as HTMLElement | null

  if (
    tailBlock?.nodeType === Node.ELEMENT_NODE &&
    (tailBlock.tagName === 'DIV' || tailBlock.tagName === 'P') &&
    isBlankNode(tailBlock)
  ) {
    editor.removeChild(tailBlock)
  }

  // Unwrap a lone block wrapper back to inline content.
  if (editor.childNodes.length === 1 && editor.firstChild?.nodeType === Node.ELEMENT_NODE) {
    const wrapper = editor.firstChild as HTMLElement

    if ((wrapper.tagName === 'DIV' || wrapper.tagName === 'P') && wrapper.dataset.slot !== RICH_INPUT_SLOT) {
      editor.replaceChildren(...Array.from(wrapper.childNodes))
    }
  }

  // A trailing <br> right after a chip / only whitespace is a phantom line.
  const last = editor.lastChild

  if (last?.nodeName === 'BR') {
    let prev: ChildNode | null = last.previousSibling

    while (prev?.nodeType === Node.TEXT_NODE && !(prev.textContent || '').trim()) {
      prev = prev.previousSibling
    }

    if (!prev || (prev as HTMLElement).dataset?.refText) {
      editor.removeChild(last)
    }
  }

  // ContentEditable elements with no children can visually collapse to
  // near-zero height in some browsers (especially Chromium), causing the
  // composer to appear as a tiny dot/pixel. Ensure there's always at least
  // one <br> so the element maintains intrinsic height. The CSS min-height
  // is a belt; the <br> is suspenders — together they prevent the shrink.
  if (editor.childNodes.length === 0) {
    editor.appendChild(document.createElement('br'))
  }
}
