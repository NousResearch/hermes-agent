import { session } from 'electron'

const EMBED_SESSION_PARTITION = 'persist:hermes-embed'
const EMBED_REFERER = 'https://www.youtube.com/'

const YOUTUBE_REFERER_HOST_RE =
  /(^|\.)(youtube\.com|youtube-nocookie\.com|googlevideo\.com|ytimg\.com|youtubei\.googleapis\.com)$/i

/**
 * YouTube rejects embed player config requests that arrive without a Referer
 * (player error 153). Chat embeds are plain iframes in the window session, and
 * the production renderer loads from a file:// URL, so the browser sends none.
 */
function stampEmbedRefererHeaders(
  url: string,
  headers: Record<string, string>
): Record<string, string> {
  let host = ''

  try {
    host = new URL(url).hostname
  } catch {
    host = ''
  }

  if (!YOUTUBE_REFERER_HOST_RE.test(host)) {
    return headers
  }

  if (headers.Referer || headers.referer) {
    return headers
  }

  return { ...headers, Referer: EMBED_REFERER }
}

/**
 * Compose the referer stamp onto another session listener. Electron keeps a
 * single listener per webRequest hook, so the default-session listener that
 * applies remote gateway headers must also carry the stamp. The stamp has to
 * run on the listener's RESULT (a wrapped listener reporting no header changes
 * would otherwise drop it again), while requests the stamp does not apply to
 * keep the listener's exact callback semantics — an empty requestHeaders value
 * means "replace all headers", not "no changes".
 */
function withEmbedRefererStamp(
  listener: (
    details: EmbedRequestDetails,
    callback: (result: { requestHeaders?: Record<string, string> }) => void
  ) => void
) {
  return (
    details: EmbedRequestDetails,
    callback: (result: { requestHeaders?: Record<string, string> }) => void
  ) => {
    listener(details, (result) => {
      if (result.requestHeaders) {
        callback({ requestHeaders: stampEmbedRefererHeaders(details.url, result.requestHeaders) })

        return
      }

      const base = details.requestHeaders ?? {}
      const stamped = stampEmbedRefererHeaders(details.url, base)

      if (stamped === base) {
        callback(result)

        return
      }

      callback({ requestHeaders: stamped })
    })
  }
}

interface EmbedRequestDetails {
  url: string
  requestHeaders?: Record<string, string>
}

interface EmbedSession {
  webRequest: {
    onBeforeSendHeaders: (
      listener: (
        details: EmbedRequestDetails,
        callback: (result: { requestHeaders?: Record<string, string> }) => void
      ) => void
    ) => void
  }
}

function installEmbedRefererForSession(embedSession: EmbedSession | undefined) {
  if (!embedSession) {
    return
  }

  embedSession.webRequest.onBeforeSendHeaders((details, callback) => {
    callback({ requestHeaders: stampEmbedRefererHeaders(details.url, details.requestHeaders ?? {}) })
  })
}

/** Stamp Referer on YouTube requests in the embed webview partition. */
function installEmbedReferer() {
  try {
    installEmbedRefererForSession(session.fromPartition(EMBED_SESSION_PARTITION))
  } catch {
    // Non-fatal: embeds still render; YouTube may show referer errors.
  }
}

export { installEmbedReferer, stampEmbedRefererHeaders, withEmbedRefererStamp }
