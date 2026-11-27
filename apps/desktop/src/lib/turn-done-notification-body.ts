// Native turn-done notification body (#88488).
// Prefer the assistant reply. If that text is missing, use a generic
// completion phrase — never the session title (usually the user's first
// message). The title-in-the-chain behavior was reported from the 0.20.0
// Studio bundle (`dist/client/assets/js/chat-*.js`), which the issue quotes
// from minified release output; this app's own chain never read the title —
// it was `text.slice(0, 140) || i18n fallback`, whose only real defect was
// the empty body when the reply was missing and the fallback shipped blank.
// The title guard test below keeps the title from ever re-entering the chain.

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
