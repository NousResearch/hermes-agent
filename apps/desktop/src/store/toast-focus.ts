// In-app toast click → open the session the toast is about. Mirrors the
// OS-notification click path (electron main → 'hermes:focus-session' →
// use-desktop-integrations), which lives behind a router handle this leaf
// store cannot reach. The wiring that owns the router installs the opener at
// boot; until then clicks are no-ops. Own leaf module so `store/notifications`
// (consumer) and the app wiring (producer) never import each other.

let openToastSession: ((sessionId: string) => void) | null = null

export function setToastSessionOpener(opener: ((sessionId: string) => void) | null): void {
  openToastSession = opener
}

export function focusToastSession(sessionId: string | undefined): void {
  if (sessionId) {
    openToastSession?.(sessionId)
  }
}
