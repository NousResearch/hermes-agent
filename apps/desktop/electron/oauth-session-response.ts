import { Buffer } from 'node:buffer'

import { httpStatusError, readApiJsonResponseWithByteLimit } from './api-transport'

// Response-stream extraction based on Bartok9's #99135 (original issue #72530).
export interface OauthResponseLike {
  on(event: 'data', cb: (chunk: Buffer) => void): void
  on(event: 'error', cb: (error: Error) => void): void
  on(event: 'end', cb: () => void): void
  statusCode?: number
  headers: Record<string, string | string[] | undefined>
}

export interface WireOauthResponseOptions {
  abort?: () => void
  method?: string
  path?: string
  url: string
  isTimedOut: () => boolean
  clearTimer: () => void
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

export function wireOauthSessionResponse(res: OauthResponseLike, opts: WireOauthResponseOptions): void {
  const { abort, method = 'GET', path, url, isTimedOut, clearTimer, resolve, reject } = opts
  let settled = false

  readApiJsonResponseWithByteLimit(res, { abort, method, path, url }).then(
    text => {
      if (settled || isTimedOut()) {
        return
      }

      settled = true
      clearTimer()
      const statusCode = res.statusCode || 500

      if (statusCode >= 400) {
        reject(httpStatusError(statusCode, text))

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
    },
    error => {
      if (settled || isTimedOut()) {
        return
      }

      settled = true
      clearTimer()
      reject(error)
    }
  )
}
