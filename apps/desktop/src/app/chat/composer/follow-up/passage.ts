/**
 * Turning a live transcript selection into a follow-up passage.
 *
 * DOM-only and React-free on purpose: the pill owns the gesture, this owns the
 * rule for what counts as a quotable selection and what the passage becomes,
 * so both halves stay readable on their own.
 */

import { normalizeFollowUpPassage } from '@/store/composer'

/**
 * Surfaces inside the transcript that must not lose text to a quote: the
 * inline message editor types into one, and the controls act on a click.
 * Quoting out of either would also fight that surface for the caret.
 */
const NOT_A_PASSAGE = '[contenteditable="true"], a, button, input, select, textarea'

/**
 * The message roots a passage may come from: the bubbles' own declared slot
 * ids, not the `data-role` values they happen to carry. The slot is the
 * attribute the emitting component writes for itself, so the contract lives
 * with the emitters (`data-slot="aui_*_message-root"`, greppable in one search)
 * instead of in a second copy of their role vocabulary.
 */
const MESSAGE_ROOT = '[data-slot="aui_assistant-message-root"], [data-slot="aui_user-message-root"]'

export interface FollowUpCapture {
  /** The passage as it will be sent — normalized once, here. */
  passage: string
  /** Which side of the transcript it came from: the card's attribution. */
  source: 'assistant' | 'user'
}

function elementOf(node: Node | null | undefined): Element | null {
  return node instanceof Element ? node : (node?.parentElement ?? null)
}

function messageRootOf(node: Node | null | undefined): HTMLElement | null {
  return elementOf(node)?.closest<HTMLElement>(MESSAGE_ROOT) ?? null
}

const isInControl = (node: Node | null | undefined): boolean => Boolean(elementOf(node)?.closest(NOT_A_PASSAGE))

/**
 * The capture a selection represents, or null when it is not a passage the
 * reader picked out of THIS transcript:
 *
 *   - collapsed — a caret is not a quote
 *   - reaching outside the viewport — the composer's own draft, a sidebar row,
 *     or the sibling pane's transcript
 *   - spanning two messages — no single side to attribute it to
 *   - inside a control that owns its selection
 */
export function captureFollowUpSelection(
  selection: Selection | null | undefined,
  viewport: Element | null
): FollowUpCapture | null {
  if (!viewport || !selection || selection.isCollapsed || selection.rangeCount === 0) {
    return null
  }

  const { anchorNode, focusNode } = selection

  if (!anchorNode || !focusNode || !viewport.contains(anchorNode) || !viewport.contains(focusNode)) {
    return null
  }

  const anchorRoot = messageRootOf(anchorNode)

  if (!anchorRoot || anchorRoot !== messageRootOf(focusNode) || isInControl(anchorNode) || isInControl(focusNode)) {
    return null
  }

  const passage = normalizeFollowUpPassage(selection.toString())

  return passage
    ? { passage, source: anchorRoot.dataset.slot === 'aui_user-message-root' ? 'user' : 'assistant' }
    : null
}
