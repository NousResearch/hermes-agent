/**
 * HTTP(S) targets for the in-app preview pane.
 *
 * The pane is the desktop's browser: a public page the user asked to open
 * has to navigate. Loopback stays special only for the 0.0.0.0 bind address,
 * which the webview cannot load — rewrite that host to 127.0.0.1. Rejecting
 * every other host made `preview.open` succeed while the pane stayed blank.
 */

export interface PreviewHttpUrlTarget {
  kind: 'url'
  label: string
  source: string
  url: string
}

export function previewHttpUrlTarget(rawTarget: string): PreviewHttpUrlTarget | null {
  const raw = String(rawTarget || '').trim()

  if (!raw) {
    return null
  }

  let url: URL

  try {
    url = new URL(raw)
  } catch {
    return null
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return null
  }

  if (url.hostname === '0.0.0.0') {
    url.hostname = '127.0.0.1'
  }

  return {
    kind: 'url',
    label: `${url.host}${url.pathname === '/' ? '' : url.pathname}`,
    source: raw,
    url: url.toString()
  }
}
