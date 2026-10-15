/**
 * Exactly one GET through Electron's `net.request`, never following a
 * redirect. `net.fetch(url, { redirect: 'manual' })` cannot be used for this:
 * Electron rejects it with "Redirect was cancelled" and the `Location` is lost,
 * so the favicon ladder re-admits every hop through `safe-http-redirects.ts`
 * with this instead.
 */

import type { EventEmitter } from 'node:events'

export interface MetadataHop {
  /** Final-response body; null for a redirect, a failure, or an over-budget body with `overflow: 'reject'`. */
  body: Buffer | null
  contentType: string
  redirectUrl: string
  statusCode: number
}

interface HopResponse extends EventEmitter {
  headers: Record<string, string | string[]>
  statusCode: number
}

interface HopRequest extends EventEmitter {
  abort(): void
  end(): void
  setHeader(name: string, value: string): void
}

export type MetadataRequestFn = (options: {
  method: 'GET'
  redirect: 'manual'
  url: string
  useSessionCookies: false
}) => HopRequest

const FAILED: MetadataHop = { body: null, contentType: '', redirectUrl: '', statusCode: 0 }

function headerValue(headers: Record<string, string | string[]>, name: string): string {
  const value = headers[name] ?? headers[name.toLowerCase()]

  return String(Array.isArray(value) ? (value[0] ?? '') : (value ?? ''))
}

export function metadataRequestOnce(
  request: MetadataRequestFn,
  url: string,
  options: { headers: Record<string, string>; maxBytes: number; overflow: 'reject' | 'truncate'; timeoutMs: number }
): Promise<MetadataHop> {
  return new Promise(resolve => {
    let settled = false
    let req: HopRequest

    const settle = (hop: MetadataHop) => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      resolve(hop)
    }

    const stop = (hop: MetadataHop) => {
      settle(hop)

      try {
        req.abort()
      } catch {
        // Already finished or torn down.
      }
    }

    const timer = setTimeout(() => stop(FAILED), Math.max(1, options.timeoutMs))

    try {
      req = request({ method: 'GET', redirect: 'manual', url, useSessionCookies: false })

      for (const [name, value] of Object.entries(options.headers)) {
        req.setHeader(name, value)
      }
    } catch {
      return settle(FAILED)
    }

    // Report the hop and cancel it; the caller's ladder decides whether the
    // Location may be dialed at all.
    req.on('redirect', (statusCode: number, _method: string, redirectUrl: string) =>
      stop({ body: null, contentType: '', redirectUrl: String(redirectUrl || ''), statusCode })
    )
    req.on('error', () => settle(FAILED))
    req.on('response', (response: HopResponse) => {
      const chunks: Buffer[] = []
      let bytes = 0
      const base = { contentType: headerValue(response.headers, 'content-type'), redirectUrl: '' }

      response.on('data', (chunk: Buffer) => {
        if (settled) {
          return
        }

        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)

        if (bytes + buffer.length > options.maxBytes) {
          if (options.overflow === 'reject') {
            return stop({ ...base, body: null, statusCode: response.statusCode })
          }

          chunks.push(buffer.subarray(0, options.maxBytes - bytes))

          return stop({ ...base, body: Buffer.concat(chunks), statusCode: response.statusCode })
        }

        chunks.push(buffer)
        bytes += buffer.length
      })
      response.on('end', () => settle({ ...base, body: Buffer.concat(chunks), statusCode: response.statusCode }))
      response.on('error', () => settle(FAILED))
    })
    req.end()
  })
}
