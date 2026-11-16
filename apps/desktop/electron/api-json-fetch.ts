/**
 * The main process's JSON-over-HTTP read core, extracted from main.ts so the
 * conditional-GET seam (C3 desktop half, t_5c4b52bf) is unit-testable without
 * Electron. main.ts keeps the `fetchJson` name and wires its own dependencies
 * (remote headers, native gateway grants, multipart assembly) into `deps`.
 *
 * Two optional behaviours, both OFF by default (default-off means every
 * existing caller is byte-identical):
 *
 * - `ifNoneMatch` sends the value as the `If-None-Match` request header.
 * - `conditional` changes the SUCCESS shape: the promise resolves with
 *   `{ data, status, etag }` instead of the parsed body, and a 304 is a
 *   success — `{ data: null, status: 304, etag }` with NO body read and NO
 *   JSON parse. Failures (4xx/5xx), retries, timeouts and auth headers are
 *   untouched: a 304 resolves, so the retry policy never sees it, and the
 *   only header added is the validator itself.
 */

import http from 'node:http'
import https from 'node:https'

import { htmlResponseError, httpStatusError, jsonAgentFor, withRetry } from './api-transport'

export interface ApiJsonFetchOptions {
  method?: string
  body?: unknown
  upload?: { filename: string; contentType?: string; bytes: ArrayBuffer }
  timeoutMs?: number
  headers?: Record<string, string>
  bearer?: string
  gatewayDescriptor?: unknown
  /** Sent verbatim as `If-None-Match`; meaningful with `conditional`. */
  ifNoneMatch?: string
  /** Resolve `{ data, status, etag }` and accept a 304 Not Modified. */
  conditional?: boolean
}

export interface ApiJsonFetchResult {
  data: unknown
  status: number
  etag: string | null
}

export interface ApiJsonFetchDeps {
  defaultTimeoutMs: number
  resolveTimeout: (timeoutMs: number | undefined, fallback: number) => number
  remoteHeaders: (url: string) => Record<string, string>
  multipart: (upload: unknown) => { body: Buffer; contentType: string }
  nativeGatewayHeaders?: (descriptor: unknown, url: string, method: string, body: unknown) => Promise<Record<string, string> | null>
}

function etagHeader(res: http.IncomingMessage): string | null {
  const value = res.headers.etag

  return typeof value === 'string' && value ? value : null
}

export function apiJsonFetch(
  url: string,
  token: string | null,
  options: ApiJsonFetchOptions = {},
  deps: ApiJsonFetchDeps
): Promise<any> {
  // Retry policy lives in api-transport.ts: idempotent verbs retry on any
  // transient transport error; POST/PUT/DELETE only when the request provably
  // never reached the server (see shouldRetryRequest) — never double-submit.
  return withRetry(
    async (requestState: any) => {
      // Mint inside each retry: a grant authorizes exactly one HTTP request.
      const nativeHeaders = options.gatewayDescriptor && deps.nativeGatewayHeaders
        ? await deps.nativeGatewayHeaders(options.gatewayDescriptor, url, options.method || 'GET', options.upload ? undefined : options.body)
        : null

      return new Promise((resolve, reject) => {
        const { body, contentType } = options.upload
          ? deps.multipart(options.upload)
          : {
              body: options.body === undefined ? undefined : Buffer.from(JSON.stringify(options.body)),
              contentType: 'application/json'
            }

        const parsed = new URL(url)
        const client = parsed.protocol === 'https:' ? https : http
        const agent = jsonAgentFor(parsed.protocol)
        const timeoutMs = deps.resolveTimeout(options.timeoutMs, deps.defaultTimeoutMs)

        if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
          reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))

          return
        }

        // Node coerces a null header value to the literal "null" on the wire
        // (the OAuth bearer leg passes token=null there); undefined throws
        // ERR_HTTP_INVALID_HEADER_VALUE. OutgoingHttpHeaders models neither,
        // so the object is typed one step wider and cast at the boundary —
        // the runtime object is unchanged byte-for-byte.
        const headers: Record<string, string | null> = {
          ...deps.remoteHeaders(url),
          ...(options.headers || {}),
          'Content-Type': contentType,
          ...(nativeHeaders || { 'X-Hermes-Session-Token': token }),
          ...(options.ifNoneMatch ? { 'If-None-Match': options.ifNoneMatch } : {}),
          // RFC 8252 native flow authenticates the gated gateway with a bearer
          // token instead of the loopback session-token header. When
          // ``options.bearer`` is set we put the bearer token in the Authorization header
          // the gateway's OAuth gate verifies it via the provider stack with
          // no cookie involved.
          ...(options.bearer ? { Authorization: `Bearer ${options.bearer}` } : {}),
          ...(body ? { 'Content-Length': String(body.length) } : {})
        }

        const req = client.request(
          parsed,
          {
            agent,
            method: options.method || 'GET',
            headers: headers as unknown as http.OutgoingHttpHeaders
          },
          res => {
            const chunks: Buffer[] = []
            res.on('error', reject)
            res.on('data', chunk => chunks.push(chunk))
            res.on('end', () => {
              const status = res.statusCode || 500

              if (status >= 400) {
                const text = Buffer.concat(chunks).toString('utf8')

                reject(httpStatusError(status, text, res.statusMessage))

                return
              }

              // A conditional read revalidating an unchanged list: the 304 is
              // the success. Its body is empty by contract (RFC 9110 §15.4.5) —
              // nothing to read, nothing to parse, the caller keeps its rows.
              if (status === 304 && options.conditional) {
                resolve({ data: null, status: 304, etag: etagHeader(res) })

                return
              }

              // http.request never follows redirects, so any 3xx -- with an HTML
              // login page, an empty body, whatever -- is the request bouncing
              // off a proxy or a scheme/slash mismatch, never JSON.
              if (status >= 300) {
                reject(htmlResponseError(url, res.statusCode, res.headers.location))

                return
              }

              const text = Buffer.concat(chunks).toString('utf8')

              if (!text) {
                resolve(options.conditional ? { data: null, status, etag: etagHeader(res) } : null)

                return
              }

              // A 2xx response whose body is HTML means the request fell through
              // to the SPA index.html (e.g. an unregistered /api path). JSON.parse
              // would throw an opaque `Unexpected token '<'` here, so surface a
              // clear diagnostic with the offending URL instead.
              const looksHtml = /^\s*<(?:!doctype|html)/i.test(text)
              const responseContentType = String(res.headers['content-type'] || '')

              if (looksHtml || responseContentType.includes('text/html')) {
                reject(htmlResponseError(url, res.statusCode))

                return
              }

              let parsedBody: unknown

              try {
                parsedBody = JSON.parse(text)
              } catch {
                reject(new Error(`Invalid JSON from ${url} (status ${res.statusCode}): ${text.slice(0, 200)}`))

                return
              }

              resolve(options.conditional ? { data: parsedBody, status, etag: etagHeader(res) } : parsedBody)
            })
          }
        )

        req.on('error', reject)
        req.setTimeout(timeoutMs, () => {
          req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
        })

        // From here the request goes on the wire: a later transport error can no
        // longer prove the server didn't process it, so non-idempotent verbs must
        // not be retried past this point.
        requestState.bodySent = true

        if (body) {
          req.write(body)
        }

        req.end()
      })
    },
    { method: options.method || 'GET' }
  )
}
