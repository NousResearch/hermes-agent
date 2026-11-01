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

/**
 * Whether a completed turn should raise the native turn-done notification at all.
 * A turn that settles with NO visible text is a quiet/control-only completion (the
 * gateway blanks intentional-silence machinery turns — internal notifications,
 * background-completion wakes — before the frame reaches the client). Notifying
 * "Message complete." for those is noise the user explicitly asked to remove;
 * every genuine reply, alert or recap carries text and still notifies.
 */
export function shouldNotifyTurnDone(content: string | null | undefined): boolean {
  const text = (content ?? '').trim()

  // Older gateways can forward the bare control token even on a user-triggered
  // silence test. Match the whole token only; prose mentioning it is real content.
  return Boolean(text) && !/^\[SILENT\]$/i.test(text)
}

export function turnDoneNotificationBody(
  content: string | null | undefined,
  fallback: string = TURN_DONE_BODY_FALLBACK
): string {
  const reply = (content ?? '').trim()
  const source = reply || fallback.trim() || TURN_DONE_BODY_FALLBACK

  return source.length <= TURN_DONE_BODY_MAX ? source : source.slice(0, TURN_DONE_BODY_MAX)
}
