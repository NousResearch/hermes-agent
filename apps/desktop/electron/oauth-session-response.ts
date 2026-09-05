/**
 * Response-stream handling for fetchJsonViaOauthSession (#72530).
 *
 * Extracted from main.ts as a pure, injectable unit so the body-error path can
 * be behavior-tested without importing the electron main module or opening a
 * real socket. The key invariant: post-response loader errors (e.g.
 * net::ERR_CONTENT_LENGTH_MISMATCH when the body is truncated after headers)
 * are emitted on the IncomingMessage, not on the ClientRequest — so we must
 * listen for `error` on the response, mirroring fetchJson/fetchPublicJson.
 */

export interface OauthResponseLike {
  on(event: 'data', cb: (chunk: unknown) => void): void
  on(event: 'error', cb: (error: Error) => void): void
  on(event: 'end', cb: () => void): void
  statusCode?: number
  headers: Record<string, string | string[] | undefined>
}

export interface WireOauthResponseOptions {
  url: string
  isTimedOut: () => boolean
  clearTimer: () => void
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

export function wireOauthSessionResponse(res: OauthResponseLike, opts: WireOauthResponseOptions): void {
  const { url, isTimedOut, clearTimer, resolve, reject } = opts
  const chunks: Buffer[] = []

  res.on('data', chunk => chunks.push(Buffer.from(chunk as never)))

  // Post-response loader errors (e.g. net::ERR_CONTENT_LENGTH_MISMATCH when the
  // body is truncated after headers) are emitted on the IncomingMessage, not on
  // the ClientRequest. Mirror fetchJson/fetchPublicJson. (#72530)
  res.on('error', error => {
    if (isTimedOut()) {
      return
    }

    clearTimer()
    reject(error)
  })

  res.on('end', () => {
    if (isTimedOut()) {
      return
    }

    clearTimer()
    const text = Buffer.concat(chunks).toString('utf8')
    const statusCode = res.statusCode || 500

    if (statusCode >= 400) {
      const err = new Error(`${statusCode}: ${text || ''}`) as Error & { statusCode?: number }
      err.statusCode = statusCode
      reject(err)

      return
    }

    if (!text) {
      resolve(null)

      return
    }

    const looksHtml = /^\s*<(?:!doctype|html)/i.test(text)
    const contentType = String(res.headers['content-type'] || res.headers['Content-Type'] || '')

    if (looksHtml || contentType.includes('text/html')) {
      reject(new Error(`Expected JSON from ${url} but got HTML (status ${statusCode}).`))

      return
    }

    try {
      resolve(JSON.parse(text))
    } catch {
      reject(new Error(`Invalid JSON from ${url} (status ${statusCode}): ${text.slice(0, 200)}`))
    }
  })
}
