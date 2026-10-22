import http from 'node:http'
import type { AddressInfo } from 'node:net'

// YouTube refuses to play (error 153) unless the embedding page has an http(s)
// origin it can match against the player's `origin` param, and on modern
// Electron a stamped Referer header is dropped. The packaged renderer is a
// file:// page, so it can't host the player itself. This loopback host serves
// one tiny wrapper page per video; the renderer iframes the wrapper and the
// wrapper iframes the player. Only the wrapper gets an http origin — the
// renderer, its storage and its backend connections stay on file://.

const YOUTUBE_ID_RE = /^[A-Za-z0-9_-]{11}$/
const PLAYER_ORIGIN = 'https://www.youtube-nocookie.com'
const PASSTHROUGH_PARAMS = ['modestbranding', 'rel', 'start'] as const

const ALLOW =
  'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen'

interface EmbedHost {
  close: () => Promise<void>
  origin: string
}

function wrapperHtml(playerUrl: string): string {
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="referrer" content="strict-origin-when-cross-origin"><style>html,body,iframe{margin:0;width:100%;height:100%;border:0;background:transparent;overflow:hidden}</style></head><body><iframe src="${playerUrl}" allow="${ALLOW}" allowfullscreen referrerpolicy="strict-origin-when-cross-origin" title="YouTube embed"></iframe></body></html>`
}

/** The wrapper document for a request path, or null for anything we don't serve. */
function youtubeWrapperFor(requestUrl: string, origin: string): null | string {
  // `//` and similar targets don't parse; throwing here would escape the
  // request handler as an uncaught main-process error and never answer.
  if (!URL.canParse(requestUrl, origin)) {
    return null
  }

  const url = new URL(requestUrl, origin)
  const match = /^\/youtube\/([^/]+)$/.exec(url.pathname)

  if (!match || !YOUTUBE_ID_RE.test(match[1])) {
    return null
  }

  const player = new URL(`/embed/${match[1]}`, PLAYER_ORIGIN)

  for (const key of PASSTHROUGH_PARAMS) {
    const value = url.searchParams.get(key)

    if (value && /^\d+$/.test(value)) {
      player.searchParams.set(key, value)
    }
  }

  player.searchParams.set('origin', origin)

  return wrapperHtml(player.toString().replaceAll('&', '&amp;'))
}

async function startEmbedHost(): Promise<EmbedHost> {
  let origin = ''

  const server = http.createServer((request, response) => {
    // Loopback only, and only our own Host: a page that DNS-rebinds onto this
    // port gets nothing.
    if (request.method !== 'GET' || request.headers.host !== new URL(origin).host) {
      response.writeHead(404).end()

      return
    }

    const html = youtubeWrapperFor(request.url || '/', origin)

    if (!html) {
      response.writeHead(404).end()

      return
    }

    response.writeHead(200, {
      'Cache-Control': 'no-store',
      'Content-Security-Policy': `default-src 'none'; frame-src ${PLAYER_ORIGIN}; style-src 'unsafe-inline'`,
      'Content-Type': 'text/html; charset=utf-8',
      'X-Content-Type-Options': 'nosniff'
    })
    response.end(html)
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port: 0 }, () => resolve())
  })

  origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`

  return {
    close: () => new Promise<void>(done => server.close(() => done())),
    origin
  }
}

let shared: null | Promise<EmbedHost> = null

/** Start the host on first use; later callers share it. A failed start retries next time. */
function embedHostOrigin(): Promise<string> {
  shared ??= startEmbedHost().catch(error => {
    shared = null

    throw error
  })

  return shared.then(host => host.origin)
}

export { embedHostOrigin, startEmbedHost, youtubeWrapperFor }
