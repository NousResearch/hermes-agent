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
  formatRefValue
} from '@/components/assistant-ui/directive-text'

export const RICH_INPUT_SLOT = 'composer-rich-input'

export const REF_RE = /@(file|folder|url|image|tool|line|terminal):(`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)/g

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

export function refLabel(id: string) {
  return id.split(/[\\/]/).filter(Boolean).pop() || id
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

export function refChipHtml(kind: string, rawValue: string) {
  const id = unquoteRef(rawValue)
  const text = `@${kind}:${quoteRefValue(id)}`

  return `<span contenteditable="false" data-ref-text="${escapeHtml(text)}" data-ref-id="${escapeHtml(id)}" data-ref-kind="${escapeHtml(kind)}" class="${DIRECTIVE_CHIP_CLASS}">${directiveIconSvg(kind)}<span class="truncate">${escapeHtml(refLabel(id))}</span></span>`
}

export function refChipElement(kind: string, rawValue: string) {
  const id = unquoteRef(rawValue)
  const text = `@${kind}:${quoteRefValue(id)}`
  const chip = document.createElement('span')
  const label = document.createElement('span')

  chip.contentEditable = 'false'
  chip.dataset.refText = text
  chip.dataset.refId = id
  chip.dataset.refKind = kind
  chip.className = DIRECTIVE_CHIP_CLASS
  label.className = 'truncate'
  label.textContent = refLabel(id)
  chip.append(directiveIconElement(kind), label)

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
  appendComposerContents(target, text)
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
