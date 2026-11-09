/**
 * Whether the empty-chat intro splash renders.
 *
 * The splash is the full-height empty state of a fresh chat. Two shapes count:
 * a pure view-reset draft in the main window (the `/new` route — no session
 * bound at all), and a DRAFT SESSION — a real, unlisted, never-sent session
 * as minted by the tab-strip "+" / ⌘T / ⌘N (`openNewSessionTab`). The
 * `draftSession` marker is authoritative for the second shape: the bound
 * session is selected, routed and active, so the view-reset gates must not
 * apply to it — only its own emptiness (messages) and the user's Appearance
 * toggle decide. Auxiliary windows stay scratch surfaces either way, and any
 * transcript at all means the conversation started.
 *
 * `enabled` is the user's Appearance toggle and outranks every other clause:
 * turning the splash off never depends on which window asks.
 */
export function shouldShowIntro(input: {
  activeSessionId: null | string
  auxiliaryWindow: boolean
  draftSession: boolean
  enabled: boolean
  freshDraftReady: boolean
  messagesEmpty: boolean
  primary: boolean
  routedSessionView: boolean
  selectedSessionId: null | string
}): boolean {
  return (
    input.enabled &&
    !input.auxiliaryWindow &&
    input.messagesEmpty &&
    (input.draftSession ||
      (input.primary &&
        input.freshDraftReady &&
        !input.routedSessionView &&
        !input.selectedSessionId &&
        !input.activeSessionId))
  )
}
