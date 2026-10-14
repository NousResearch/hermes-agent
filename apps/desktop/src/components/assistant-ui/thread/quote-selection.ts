/**
 * Reading a quotable passage out of the transcript.
 *
 * DOM-only and React-free on purpose: the floating action owns the gesture,
 * this owns the rule for what counts as a quotable selection and what the
 * passage becomes, so both halves stay readable (and testable) on their own.
 */

/**
 * Surfaces inside the transcript that must not lose text to a quote: the inline
 * message editor types into one, and the controls act on a click. Quoting out of
 * either would also fight that surface for the caret.
 */
const NOT_A_PASSAGE = '[contenteditable="true"], a, button, input, select, textarea'

/**
 * The message roots a passage may come from — the bubbles' own declared slot
 * ids, so the contract lives with the emitters instead of a second copy of their
 * role vocabulary.
 */
const MESSAGE_ROOT = '[data-slot="aui_assistant-message-root"], [data-slot="aui_user-message-root"]'

/** The scroll container a passage has to live in — a selection in the composer,
 *  a sidebar row, or the sibling pane's transcript is not a passage. */
const THREAD_VIEWPORT = '[data-slot="aui_thread-viewport"]'

export type QuoteSource = 'assistant' | 'user'

export interface QuoteCapture {
  /** The passage exactly as it will be sent — normalized once, here. */
  text: string
  /** Which side of the transcript it came from. */
  source: QuoteSource
}

/**
 * Collapse what the browser adds to a selection: CRLF, the trailing spaces a
 * wrapped line carries, the blank line a triple-click leaves behind.
 */
export function normalizeQuoteText(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map(line => line.replace(/\s+$/, ''))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

/**
 * The passage as it reaches the agent: a bracket marker plus a markdown
 * blockquote, so the follow-up question is visibly about the quote rather than
 * the whole reply. The marker stays English like the gateway's own
 * `[Replying to your previous message: …]` inbound prefix — it is model-facing
 * scaffolding, not UI copy.
 */
export function quoteBlock(capture: QuoteCapture): string {
  const marker = capture.source === 'user' ? '[Quoted from my earlier message]' : '[Quoted from your previous message]'

  const body = capture.text
    .split('\n')
    .map(line => `> ${line}`.trimEnd())
    .join('\n')

  return `${marker}\n${body}`
}

function elementOf(node: Node | null | undefined): Element | null {
  return node instanceof Element ? node : (node?.parentElement ?? null)
}

const isInControl = (node: Node | null | undefined): boolean => Boolean(elementOf(node)?.closest(NOT_A_PASSAGE))

/**
 * The capture a selection represents, or null when it is not a passage the
 * reader picked out of THIS transcript:
 *
 *   - collapsed — a caret is not a quote
 *   - outside a thread viewport — the composer's own draft, a sidebar row
 *   - reaching outside the message it started in — no single side to attribute
 *   - inside a control that owns its selection
 */
export function captureQuoteSelection(selection: Selection | null | undefined): QuoteCapture | null {
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return null
  }

  const { anchorNode, focusNode } = selection

  if (!anchorNode || !focusNode) {
    return null
  }

  const anchorRoot = elementOf(anchorNode)?.closest<HTMLElement>(MESSAGE_ROOT) ?? null
  const focusRoot = elementOf(focusNode)?.closest<HTMLElement>(MESSAGE_ROOT) ?? null

  if (!anchorRoot || anchorRoot !== focusRoot || !anchorRoot.closest(THREAD_VIEWPORT)) {
    return null
  }

  if (isInControl(anchorNode) || isInControl(focusNode)) {
    return null
  }

  const text = normalizeQuoteText(selection.toString())

  return text ? { text, source: anchorRoot.dataset.slot === 'aui_user-message-root' ? 'user' : 'assistant' } : null
}

/**
 * Where to put the action: the top of the first rect the selection covers.
 * Returns null when the range has no box (a selection inside a collapsed or
 * hidden element), which callers read as "no anchor, stay hidden".
 */
export function selectionAnchorRect(selection: Selection | null | undefined): DOMRect | null {
  if (!selection || selection.rangeCount === 0) {
    return null
  }

  const range = selection.getRangeAt(0)

  return range.getClientRects()[0] ?? null
}
