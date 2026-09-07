// URL helpers for the hosted pen.dev embed. Kept Electron-free so the
// "stay on /new?embed" contract can be unit-tested — Pencil's /new page
// prerenders with an empty query and a client effect then does
// `router.replace(/new?d=<uuid>)`, which drops `embed` and paints the
// full web app (local files, Sign In, Goodies).

/** Same-origin as the hosted editor (path / query may differ). */
export function isPenWebUrl(url: string, webEditorUrl: string): boolean {
  try {
    return new URL(url).origin === new URL(webEditorUrl).origin
  } catch {
    return false
  }
}

/** Guarantee the editor URL carries `embed` so Pencil's page can see it. */
export function ensurePenEmbedUrl(url: string): string {
  try {
    const parsed = new URL(url)

    if (!parsed.searchParams.has('embed')) {
      parsed.searchParams.set('embed', '')
    }

    return parsed.toString()
  } catch {
    return url
  }
}

/** Pencil navigated to /new without `embed` — we left embed mode. */
export function penEmbedDropped(currentUrl: string, editorUrl: string): boolean {
  try {
    const current = new URL(currentUrl)
    const editor = new URL(editorUrl)

    if (current.origin !== editor.origin) {
      return false
    }

    const path = current.pathname.replace(/\/$/, '') || '/'

    return path === '/new' && !current.searchParams.has('embed')
  } catch {
    return false
  }
}

/** Put `embed` back, keep Pencil's minted `d` so the replace effect stays quiet. */
export function restorePenEmbedUrl(currentUrl: string, editorUrl: string): string {
  const next = new URL(ensurePenEmbedUrl(editorUrl))

  try {
    const current = new URL(currentUrl)
    const documentId = current.searchParams.get('d')

    if (documentId && !next.searchParams.get('d')) {
      next.searchParams.set('d', documentId)
    }
  } catch {
    // currentUrl was unparseable — return the editor URL with embed only.
  }

  return next.toString()
}
