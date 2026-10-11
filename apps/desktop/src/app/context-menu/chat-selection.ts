export const CHAT_MESSAGE_SELECTOR = '[data-slot="aui_assistant-message-root"], [data-slot="aui_user-message-root"]'
const EDITABLE_SELECTOR = 'input, textarea, [contenteditable]:not([contenteditable="false"])'
export const MAX_CHAT_SELECTION_CHARS = 16_000

export interface ChatSelection {
  text: string
  sessionId: string
  message: Element
}

/** A selection belongs to exactly one non-editable message and its rendered session. */
export function captureChatSelection(element: Element | null, selection = window.getSelection()): ChatSelection | null {
  if (
    !element ||
    !selection ||
    selection.rangeCount !== 1 ||
    selection.isCollapsed ||
    element.closest(EDITABLE_SELECTOR)
  ) {
    return null
  }

  const elementFor = (node: Node | null) =>
    node?.nodeType === Node.ELEMENT_NODE ? (node as Element) : node?.parentElement

  const anchor = elementFor(selection.anchorNode)
  const focus = elementFor(selection.focusNode)
  const message = anchor?.closest(CHAT_MESSAGE_SELECTOR)
  const text = selection.toString().trim()
  const sessionId = message?.closest('[data-selection-session-id]')?.getAttribute('data-selection-session-id')

  if (
    !message ||
    message !== focus?.closest(CHAT_MESSAGE_SELECTOR) ||
    !message.contains(element) ||
    anchor?.closest(EDITABLE_SELECTOR) ||
    focus?.closest(EDITABLE_SELECTOR) ||
    !text ||
    text.length > MAX_CHAT_SELECTION_CHARS ||
    !sessionId
  ) {
    return null
  }

  return { message, sessionId, text }
}

export function chatSelectionIsCurrent(expected: ChatSelection): boolean {
  if (!expected.message.isConnected) {
    return false
  }

  const current = captureChatSelection(expected.message)

  return (
    !!current &&
    current.message === expected.message &&
    current.text === expected.text &&
    current.sessionId === expected.sessionId
  )
}
