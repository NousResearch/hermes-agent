import { session } from 'electron'

const EMBED_SESSION_PARTITION = 'persist:hermes-embed'
const EMBED_REFERER = 'https://com.nousresearch.hermes/'

const YOUTUBE_REFERER_HOST_RE =
  /(^|\.)(youtube\.com|youtube-nocookie\.com|googlevideo\.com|ytimg\.com|youtubei\.googleapis\.com)$/i

function installEmbedRefererForSession(embedSession) {
  if (!embedSession) {
    return
  }

  embedSession.webRequest.onBeforeSendHeaders((details, callback) => {
    let host = ''

    try {
      host = new URL(details.url).hostname
    } catch {
      host = ''
    }

    if (!YOUTUBE_REFERER_HOST_RE.test(host)) {
      callback({ requestHeaders: details.requestHeaders })

      return
    }

    const headers = { ...details.requestHeaders }

    if (!headers.Referer && !headers.referer) {
      headers.Referer = EMBED_REFERER
    }

    callback({ requestHeaders: headers })
  })
}

/** Stamp Referer on YouTube requests used by plain iframes and embed webviews. */
function installEmbedReferer() {
  const sessionFactories = [
    () => session.defaultSession,
    () => session.fromPartition(EMBED_SESSION_PARTITION)
  ]

  for (const getSession of sessionFactories) {
    try {
      installEmbedRefererForSession(getSession())
    } catch {
      // Non-fatal: one failed session must not disable embeds in the other.
    }
  }
}

export { installEmbedReferer }
