// Admission for URLs handed up by the preview guest preload
// (electron/preview-guest-preload.ts, `preview-open-external` channel).
//
// A guest page chose this URL, not the user typing it, so the scheme set is
// strictly the web subset of what main's `openExternalUrl` accepts. `file:`
// stays out on purpose: main's opener does open local files, but that
// capability belongs to host-driven surfaces (the artifacts panel), and an
// untrusted preview page must never reach `shell.openPath` through us.

import { isLoopbackPreviewUrl } from '@/lib/local-preview'

export const PREVIEW_EXTERNAL_CHANNEL = 'preview-open-external'

export function admitPreviewExternalUrl(rawUrl: string): boolean {
  const raw = String(rawUrl ?? '').trim()

  if (!raw) {
    return false
  }

  try {
    const { protocol } = new URL(raw)

    return protocol === 'https:' || protocol === 'http:'
  } catch {
    return false
  }
}

export type PreviewGuestLinkRoute = 'deny' | 'external' | 'in-app'

/**
 * Where a guest's `_blank` anchor goes (the URL has already crossed the
 * preload's trusted-click gate).
 *
 * - A page of the user's own machine (a dev-server preview on loopback, a
 *   local file) is a preview OF something the agent built; its outbound links
 *   (Streamlit's "Ask Google", #112941) belong in the OS browser.
 * - A remote web page is the in-app Browser being used as a browser, and a
 *   browser's `_blank` stays in the browser. Sending it to the OS browser
 *   pulls the user out of the pane they are working in — and out from under
 *   an agent driving it, which then reads a page that never changed.
 *
 * Either way the URL is admitted by `admitPreviewExternalUrl` first, so this
 * can only ever narrow where an admitted URL opens, never widen what is
 * admitted.
 */
export function routePreviewGuestLink(guestUrl: string, linkUrl: string): PreviewGuestLinkRoute {
  if (!admitPreviewExternalUrl(linkUrl)) {
    return 'deny'
  }

  try {
    const { protocol } = new URL(guestUrl)

    if ((protocol === 'http:' || protocol === 'https:') && !isLoopbackPreviewUrl(guestUrl)) {
      return 'in-app'
    }
  } catch {
    // An unreadable guest URL keeps the pre-existing OS-browser handoff.
  }

  return 'external'
}
