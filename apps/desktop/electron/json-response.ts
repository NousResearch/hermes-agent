import type { IncomingMessage, RequestOptions } from 'node:http'
import type http from 'node:http'
import { gunzipSync } from 'node:zlib'

import { htmlResponseError, httpStatusError } from './api-transport'

// Over loopback the transfer costs nothing, so compressing would only spend server CPU.
const LOCAL_HOSTNAMES = new Set(['localhost', '127.0.0.1', '[::1]', '0.0.0.0', '[::]'])

// A remote backend chooses the gzip body; cap what it can inflate to, so a gzip bomb rejects
// instead of exhausting the main process. A 500-message transcript page decodes to ~1.6 MB.
export const JSON_RESPONSE_MAX_DECODED_BYTES = 64 * 1024 * 1024

// A remote backend is asked for gzip whatever its URL: an SSH backend is dialled through a
// 127.0.0.1 port forward. Without a descriptor, the URL decides.
export function asksForGzip(url: URL, remoteBackend = false): boolean {
  const host = url.hostname.toLowerCase()

  const local =
    LOCAL_HOSTNAMES.has(host) || host.startsWith('127.') || host.startsWith('[::ffff:7f') || host.endsWith('.localhost')

  return remoteBackend || !local
}

export function decodeJsonResponseBody(
  chunks: Buffer[],
  encoding: string | string[] | undefined,
  maxDecodedBytes = JSON_RESPONSE_MAX_DECODED_BYTES
): string {
  const body = Buffer.concat(chunks)
  // An empty body (a proxy's 204/304) has nothing to gunzip; keep its status for the caller.
  const gzipped = body.length > 0 && String(encoding).trim().toLowerCase() === 'gzip'

  if (!gzipped) {
    return body.toString('utf8')
  }

  try {
    return gunzipSync(body, { maxOutputLength: maxDecodedBytes }).toString('utf8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ERR_BUFFER_TOO_LARGE') {
      throw new Error(`JSON response exceeds the ${maxDecodedBytes}-byte decoded limit`)
    }

    throw error
  }
}

export function decodeJsonResponse(
  chunks: Buffer[],
  encoding: string | string[] | undefined,
  reject: (error: Error) => void,
  onText: (text: string) => void
): void {
  try {
    onText(decodeJsonResponseBody(chunks, encoding))
  } catch (error) {
    reject(error as Error)
  }
}

// fetchJson and fetchPublicJson settle a decoded body the same way.
export function settleJsonResponse(
  url: string,
  res: IncomingMessage,
  text: string,
  resolve: (value: unknown) => void,
  reject: (error: Error) => void
): void {
  const statusCode = res.statusCode || 500

  if (statusCode >= 400) {
    reject(httpStatusError(res.statusCode, text, res.statusMessage))

    return
  }

  // http.request never follows redirects, so any 3xx -- with an HTML
  // login page, an empty body, whatever -- is the request bouncing
  // off a proxy or a scheme/slash mismatch, never JSON.
  if (statusCode >= 300) {
    reject(htmlResponseError(url, res.statusCode, res.headers.location))

    return
  }

  if (!text) {
    resolve(null)

    return
  }

  // A 2xx response whose body is HTML means the request fell through
  // to the SPA index.html (e.g. an unregistered /api path). JSON.parse
  // would throw an opaque `Unexpected token '<'` here, so surface a
  // clear diagnostic with the offending URL instead.
  const looksHtml = /^\s*<(?:!doctype|html)/i.test(text)
  const contentType = String(res.headers['content-type'] || '')

  if (looksHtml || contentType.includes('text/html')) {
    reject(htmlResponseError(url, res.statusCode))

    return
  }

  try {
    resolve(JSON.parse(text))
  } catch {
    reject(new Error(`Invalid JSON from ${url} (status ${res.statusCode}): ${text.slice(0, 200)}`))
  }
}

// Chromium's electron.net already decodes responses; this is for Node HTTP only.
export function requestJsonResponse(
  client: Pick<typeof http, 'request'>,
  url: URL,
  options: RequestOptions,
  onResponse: (response: IncomingMessage) => void,
  remoteBackend = false
) {
  const headers = asksForGzip(url, remoteBackend) ? { ...options.headers, 'Accept-Encoding': 'gzip' } : options.headers

  return client.request(url, { ...options, headers }, onResponse)
}
