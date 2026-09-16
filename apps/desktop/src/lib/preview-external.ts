// Admission for URLs handed up by the preview guest preload
// (electron/preview-guest-preload.ts, `preview-open-external` channel).
//
// A guest page chose this URL, not the user typing it, so the scheme set is
// strictly the web/mail subset of what main's `openExternalUrl` accepts.
// `file:` stays out on purpose: main's opener does open local files, but that
// capability belongs to host-driven surfaces (the artifacts panel), and an
// untrusted preview page must never reach `shell.openPath` through us.

export const PREVIEW_EXTERNAL_CHANNEL = 'preview-open-external'

export function admitPreviewExternalUrl(rawUrl: string): boolean {
  const raw = String(rawUrl ?? '').trim()

  if (!raw) {
    return false
  }

  try {
    const { protocol } = new URL(raw)

    return protocol === 'https:' || protocol === 'http:' || protocol === 'mailto:'
  } catch {
    return false
  }
}
