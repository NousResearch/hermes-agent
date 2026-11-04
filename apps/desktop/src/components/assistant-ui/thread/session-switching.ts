/**
 * `data-session-switching` — the supported hook for styling a chat switch.
 *
 * Core sets it on the chat surface root (`[data-chat-surface]`, one per pane:
 * the primary chat and every tile) while that surface is placing a transcript:
 * from the frame a routed session starts loading, or a different transcript
 * lands, until its rows are on screen and the restored scroll position has
 * settled. Theme CSS may target the attribute (hide the transcript while it is
 * set, fade it in when it goes); internal class names and the surface's inner
 * structure are not a contract.
 *
 * Several phases can hold the marker at once (the load, then the scroll
 * restore). Each holder releases only itself, and the attribute goes when the
 * last one does, so a phase ending can never clear another's mark mid-switch.
 */
export const SESSION_SWITCHING_ATTRIBUTE = 'data-session-switching'

const holders = new WeakMap<HTMLElement, Set<symbol>>()

/** Mark the chat surface containing `from` as switching; returns the release. */
export function holdSessionSwitching(from: HTMLElement | null): () => void {
  const surface = from?.closest<HTMLElement>('[data-chat-surface]')

  if (!surface) {
    return () => {}
  }

  const held = holders.get(surface) ?? new Set<symbol>()
  const token = Symbol('session-switch')
  holders.set(surface, held)
  held.add(token)
  surface.setAttribute(SESSION_SWITCHING_ATTRIBUTE, 'true')

  return () => {
    if (held.delete(token) && held.size === 0) {
      surface.removeAttribute(SESSION_SWITCHING_ATTRIBUTE)
    }
  }
}
