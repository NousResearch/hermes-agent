declare global {
  interface Window {
    /**
     * Injected by the server as `true`. The embedded TUI Chat surface
     * (`/chat`, `/api/ws`, `/api/pty`) is always enabled, so this is
     * effectively a constant; kept on `window` for any consumer that reads
     * it directly and for parity with the server's bootstrap script.
     */
    __HERMES_DASHBOARD_EMBEDDED_CHAT__?: boolean;
  }
}

/**
 * Whether the dashboard's embedded TUI Chat surface is available.
 *
 * The embedded chat (`/chat` tab, `/api/ws` + `/api/pty` WebSockets) is now
 * an unconditional part of the dashboard — the desktop app and the in-browser
 * Chat tab both depend on it — so this always returns `true`. The function is
 * retained as a stable seam so call sites don't need to change if the surface
 * ever becomes conditional again.
 */
export function isDashboardEmbeddedChatEnabled(): boolean {
  return true;
}
