import { session } from 'electron'

const EMBED_SESSION_PARTITION = 'persist:hermes-embed'
const EMBED_REFERER = 'https://www.youtube.com/'

const YOUTUBE_REFERER_HOST_RE =
  /(^|\.)(youtube\.com|youtube-nocookie\.com|googlevideo\.com|ytimg\.com|youtubei\.googleapis\.com)$/i

function withEmbedReferer(url: string, requestHeaders: Record<string, string> = {}) {
  let host = ''

  try {
    host = new URL(url).hostname
  } catch {
    host = ''
  }

  if (!YOUTUBE_REFERER_HOST_RE.test(host)) {
    return requestHeaders
  }

  if (requestHeaders.Referer || requestHeaders.referer) {
    return requestHeaders
  }

  return { ...requestHeaders, Referer: EMBED_REFERER }
}

function installEmbedRefererForSession(embedSession) {
  if (!embedSession) {
    return
  }

  embedSession.webRequest.onBeforeSendHeaders((details, callback) => {
    callback({ requestHeaders: withEmbedReferer(details.url, details.requestHeaders) })
  })
}

/**
 * Stamp Referer on YouTube requests in the embed webview partition.
 *
 * The default session also needs this (the YouTube iframe embed renders
 * there as a plain iframe, not a webview on the embed partition), but
 * Electron's webRequest.onBeforeSendHeaders only allows a single listener
 * per session — registering here would silently replace, or be replaced
 * by, main.ts's installRemoteHeaderRules() listener on the same session.
 * So the default-session stamping is composed into that listener via
 * withEmbedReferer() instead of registered here; see main.ts.
 */
function installEmbedReferer() {
  try {
    installEmbedRefererForSession(session.fromPartition(EMBED_SESSION_PARTITION))
  } catch {
    // Non-fatal: embeds still render; YouTube may show referer errors.
  }
}

export { installEmbedReferer, withEmbedReferer }
