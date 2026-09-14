/**
 * Window-open policy for the `<webview>` guests the Preview pane embeds.
 *
 * A previewed page can ask for a new window two ways: `window.open()` and an
 * `<a target="_blank">` click. Both land in Chromium's "create new window"
 * path, and for a `<webview>` that path is gated twice:
 *
 *  1. `CanCreateWindow` returns false outright when the guest's `disablePopups`
 *     preference is set — which is exactly what Electron derives from a MISSING
 *     `allowpopups` attribute on the `<webview>` element. The request dies
 *     there, before any JS handler is consulted.
 *  2. Only if popups are allowed does the guest's `setWindowOpenHandler` run
 *     and get to decide.
 *
 * That is why the Preview pane silently ate `target="_blank"` clicks (#81660):
 * gate 1 rejected them, so there was nothing left for the app to route. The
 * renderer can't fix it on its own either — the `<webview>` `new-window` DOM
 * event was removed in Electron 22, and the desktop ships Electron 40.
 *
 * So the pane now sets `allowpopups` to get the request past gate 1, and this
 * module supplies gate 2: a handler installed from the main process (via the
 * embedder's `did-attach-webview`) that ALWAYS denies the popup itself and
 * instead routes the URL — http/https navigate the Preview pane in place,
 * `mailto:` goes to the OS handler, everything else is dropped.
 *
 * Denying every popup is what keeps `allowpopups` from widening the attack
 * surface: no guest can conjure an OS window, and the schemes that reach the
 * host are limited to the two a link in a web page may legitimately use. In
 * particular `file:` is NOT forwarded — `openExternalUrl` would hand it to
 * `shell.openPath`, and a previewed page must never be able to open local
 * files through the OS.
 *
 * Navigating in place (rather than opening the OS browser) is the behaviour the
 * issue asks for: the Preview pane is the surface the user is looking at, and
 * its URL header already follows the guest via `did-navigate`.
 */

/** What to do with a new-window request coming out of a `<webview>` guest. */
export type WebviewWindowOpenDecision =
  /** Load the URL in the requesting guest itself (in-pane navigation). */
  | { action: 'navigate'; url: string }
  /** Hand the URL to the OS (`shell.openExternal`). */
  | { action: 'external'; url: string }
  /** Drop the request. */
  | { action: 'block' }

/**
 * Classify a new-window request by its URL. Pure — the routing table lives
 * here so the policy can be unit-tested without booting Electron.
 *
 * `about:blank` (a bare `window.open()`) and any unparseable or non-web scheme
 * are blocked: there is no page for the pane to show and nothing safe to hand
 * the OS.
 */
export function decideWebviewWindowOpen(rawUrl: string | null | undefined): WebviewWindowOpenDecision {
  const raw = String(rawUrl ?? '').trim()

  if (!raw) {
    return { action: 'block' }
  }

  let parsed: URL

  try {
    parsed = new URL(raw)
  } catch {
    return { action: 'block' }
  }

  if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
    return { action: 'navigate', url: parsed.toString() }
  }

  if (parsed.protocol === 'mailto:') {
    return { action: 'external', url: parsed.toString() }
  }

  return { action: 'block' }
}

/** The slice of `Electron.WebContents` this module drives on a guest. */
interface WebviewGuest {
  isDestroyed: () => boolean
  loadURL: (url: string) => Promise<void>
  setWindowOpenHandler: (handler: (details: { url: string }) => { action: 'deny' }) => void
}

/** The slice of `Electron.WebContents` this module listens on for an embedder. */
interface WebviewEmbedder {
  on: (event: 'did-attach-webview', listener: (event: unknown, guest: WebviewGuest) => void) => unknown
  off: (event: 'did-attach-webview', listener: (event: unknown, guest: WebviewGuest) => void) => unknown
}

interface WebviewWindowOpenPolicyOptions {
  /** Opens a URL with the OS handler. Same contract as main.ts's `openExternalUrl`. */
  openExternal: (url: string) => unknown
  /** Diagnostic sink for a rejected in-pane navigation. */
  log?: (message: string) => void
}

/**
 * Install {@link decideWebviewWindowOpen} on every `<webview>` that attaches to
 * `embedder`, and return an uninstall function.
 *
 * `did-attach-webview` is the supported main-process replacement for the
 * removed renderer-side `new-window` event, and it fires at attach time — before
 * the guest has loaded anything — so no click can outrun the handler.
 *
 * The handler always returns `deny`; the routing happens as a side effect.
 */
export function installWebviewWindowOpenPolicy(
  embedder: WebviewEmbedder | null | undefined,
  { openExternal, log = () => {} }: WebviewWindowOpenPolicyOptions
): () => void {
  if (!embedder) {
    return () => {}
  }

  const onAttach = (_event: unknown, guest: WebviewGuest) => {
    guest.setWindowOpenHandler(details => {
      const decision = decideWebviewWindowOpen(details?.url)

      if (decision.action === 'navigate') {
        if (!guest.isDestroyed()) {
          void Promise.resolve(guest.loadURL(decision.url)).catch(error =>
            log(`[preview] webview navigation failed: ${error instanceof Error ? error.message : String(error)}`)
          )
        }
      } else if (decision.action === 'external') {
        openExternal(decision.url)
      }

      return { action: 'deny' }
    })
  }

  embedder.on('did-attach-webview', onAttach)

  return () => {
    embedder.off('did-attach-webview', onAttach)
  }
}
