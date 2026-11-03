/**
 * Address parsing for the Browser's address bar and start page. Pure, no DOM:
 * what the user typed (or what a shortcut chip stands for) → something a
 * webview can load, or `null` when it isn't loadable.
 */

/**
 * What the user typed, as something a webview can load — or `null` if it isn't
 * loadable. A bare host is the common case in an address bar, and here it is
 * usually a dev server, so loopback gets `http` (nothing is listening on 443
 * and there's no certificate) and anything else gets `https`.
 *
 * The blocklist is deliberately narrow: `javascript:` and `data:` execute in
 * the guest partition rather than navigating it, so an address bar is not the
 * place to reach them. Everything a browser normally loads — including
 * `about:blank` and `file://` — goes through, because this IS a browser and
 * the person typing already owns the machine it runs on.
 */
export function normalizePreviewAddress(value: string): null | string {
  const address = value.trim()

  if (!address) {
    return null
  }

  // `://`, not a leading-scheme regex: `localhost:5173` and `example.com:8080`
  // both match `scheme:` and would be read as an unknown protocol instead of a
  // host and a port. `about:blank` has no `://` either, hence the explicit
  // scheme test before the host guess.
  const scheme = /^(about|blob|chrome|data|devtools|file|ftp|https?|javascript|view-source):/i.exec(address)?.[1]
  const loopback = /^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?(?:[/?#]|$)/i.test(address)
  const candidate = scheme ? address : `${loopback ? 'http' : 'https'}://${address}`

  // Script-bearing schemes run INSIDE the page the user is looking at (or mint
  // a document with an attacker-chosen body) — that's injection wearing a
  // navigation's clothes, and no browser's address bar honors it either.
  if (scheme && /^(data|javascript)$/i.test(scheme)) {
    return null
  }

  try {
    // A parse failure is the real filter for junk: `://broken`, `htp:/x`, and
    // half-typed nonsense all land here.
    new URL(candidate)

    return candidate
  } catch {
    return null
  }
}

/**
 * The dev servers the start page offers as one-click chips. Only the ports a
 * local app usually lands on: the point is to save typing an address the person
 * already knows, not to guess what is running.
 */
export const START_SHORTCUTS = ['localhost:3000', 'localhost:5173', 'localhost:8000'] as const

/** The address a start-page chip navigates to, or `null` if it isn't loadable. */
export function startShortcutAddress(shortcut: string): null | string {
  return normalizePreviewAddress(shortcut)
}
