// Native turn-done notification body (#88488).
// Prefer the assistant reply. If that text is missing, use a generic
// completion phrase — never the session title (usually the user's first
// message). The minified Studio builder was `content || session.title ||
// "Message complete."`; title in that chain made the OS toast repeat the
// question instead of the answer.

export const TURN_DONE_BODY_MAX = 140
export const TURN_DONE_BODY_FALLBACK = 'Message complete.'

export function turnDoneNotificationBody(
  content: string | null | undefined,
  fallback: string = TURN_DONE_BODY_FALLBACK
): string {
  const reply = (content ?? '').trim()
  const source = reply || fallback.trim() || TURN_DONE_BODY_FALLBACK
  return source.length <= TURN_DONE_BODY_MAX ? source : source.slice(0, TURN_DONE_BODY_MAX)
}
