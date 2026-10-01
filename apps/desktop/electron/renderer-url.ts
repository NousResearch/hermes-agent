// Same-renderer navigation guard. The pure, Electron-free piece lives here so it
// can be unit-tested (mirroring how the rest of electron/*.ts splits testable
// logic out of the main.ts monolith).

import path from 'node:path'

/**
 * True when `url` belongs to the renderer we loaded ourselves.
 *
 * Compares PARSED ORIGINS, never string prefixes. A prefix compare
 * (`url.startsWith(base)`) accepts `http://127.0.0.1:47891@attacker.example/`:
 * everything before the `@` is userinfo, so that URL's real origin is
 * `attacker.example` while still sharing a textual prefix with our base URL.
 * Letting it through `will-navigate` would load hostile content into a window
 * that carries the `hermesDesktop` preload bridge.
 */
function isRendererUrl(url: string, base: string): boolean {
  let target: URL
  let rendererBase: URL

  try {
    target = new URL(url)
    rendererBase = new URL(base)
  } catch {
    // Unparseable on either side: fail closed rather than guessing.
    return false
  }

  // `file:` URLs have an opaque ("null") origin, so origin equality is useless
  // there — two unrelated files would both compare as "null". Compare the
  // concrete path instead. Query/hash routing (`?win=secondary#/<id>`) does not
  // change `pathname`, so ordinary in-app navigation still passes.
  if (rendererBase.protocol === 'file:') {
    if (target.protocol !== 'file:') {
      return false
    }

    return path.resolve(decodeURIComponent(target.pathname)) === path.resolve(decodeURIComponent(rendererBase.pathname))
  }

  return target.origin === rendererBase.origin
}

export { isRendererUrl }
