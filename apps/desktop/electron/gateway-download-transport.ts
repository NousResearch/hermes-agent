import http from 'node:http'
import https from 'node:https'

import { downloadAgentFor, withRetry } from './api-transport'
import { DEFAULT_FETCH_TIMEOUT_MS, resolveTimeoutMs } from './hardening'
import { nativeGatewayHttpHeaders } from './local-gateway'

// Retry only the connection phase, never a save dialog or a partially saved body.
export async function downloadViaTokenToFile(url, token, ctx, finalizeGatewayDownload, options: any = {}) {
  const parsed = new URL(url)

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`)
  }

  const res: http.IncomingMessage = await withRetry(async () => {
    const headers = options.gatewayDescriptor
      ? await nativeGatewayHttpHeaders(options.gatewayDescriptor, url)
      : options.bearer ? { Authorization: `Bearer ${options.bearer}` } : { 'X-Hermes-Session-Token': token }

    return new Promise<http.IncomingMessage>((resolve, reject) => {
      const client = parsed.protocol === 'https:' ? https : http
      const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

      const req = client.request(parsed, {
        agent: downloadAgentFor(parsed.protocol), method: 'GET', headers
      }, response => {
        req.setTimeout(0)
        resolve(response)
      })

      req.on('error', reject)
      req.setTimeout(timeoutMs, () => req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`)))
      req.end()
    })
  }, { method: 'GET' })

  // Node does not follow redirects: never send a one-use grant to a new target.
  if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400) {
    res.destroy()
    throw new Error(`Unexpected download redirect (${res.statusCode})`)
  }

  return finalizeGatewayDownload(res, res.statusCode || 500, res.headers || {}, {
    ...ctx, abort: () => res.destroy()
  })
}
