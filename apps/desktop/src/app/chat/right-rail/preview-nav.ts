/**
 * Browser gestures that land on the app's own chrome — ⌘R with the address bar
 * focused, a mouse's back/forward buttons over the pane's frame.
 *
 * The interesting case is handled elsewhere: when the user is INSIDE the guest
 * page, main acts on the focused webContents directly (see
 * `commandFocusedGuest`), because a webview guest is out-of-process and nothing
 * in this renderer can see it. This registry only covers the other half — focus
 * sitting in Hermes' own DOM, where `activeElement` is authoritative.
 */

/** Marks a live browser pane so a gesture can find the one holding focus. */
export const PREVIEW_BROWSER_ATTR = 'data-preview-browser'

export interface PreviewNavHandle {
  back: () => void
  forward: () => void
  reload: () => void
}

const handles = new Map<string, PreviewNavHandle>()

/** Register a live browser pane's commands; returns an idempotent remove. */
export function registerPreviewNav(tabId: string, handle: PreviewNavHandle): () => void {
  handles.set(tabId, handle)

  return () => {
    if (handles.get(tabId) === handle) {
      handles.delete(tabId)
    }
  }
}

/** Run `command` on the browser pane holding DOM focus. False = focus is
 *  elsewhere in the app, so the caller falls back to the app-level meaning. */
export function commandFocusedPreview(command: keyof PreviewNavHandle): boolean {
  const host = document.activeElement?.closest(`[${PREVIEW_BROWSER_ATTR}]`)
  const nav = host ? handles.get(host.getAttribute(PREVIEW_BROWSER_ATTR) || '') : undefined

  nav?.[command]()

  return Boolean(nav)
}
