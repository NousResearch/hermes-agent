export interface ChatHeaderVisibilityInput {
  activeSessionId: null | string
  isRoutedSessionView: boolean
  selectedSessionId: null | string
}

export const HERMES_WINDOW_TITLE = 'Hermes'

export function shouldShowChatHeader({
  activeSessionId,
  isRoutedSessionView,
  selectedSessionId
}: ChatHeaderVisibilityInput): boolean {
  // Session pop-out windows still need the same clickable title/actions header
  // as the primary chat. Only a true empty draft has nothing meaningful to show.
  return Boolean(selectedSessionId || activeSessionId || isRoutedSessionView)
}

export function windowTitleForChat(title: string): string {
  const trimmed = title.trim()

  return trimmed && trimmed !== HERMES_WINDOW_TITLE ? `${trimmed} — ${HERMES_WINDOW_TITLE}` : HERMES_WINDOW_TITLE
}
