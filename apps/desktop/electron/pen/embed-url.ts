// URL helpers for the hosted pen.dev embed, kept Electron-free so they can be
// unit-tested. The editor keeps `embed` on its own once it has it; we only
// make sure the first load carries it.

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
