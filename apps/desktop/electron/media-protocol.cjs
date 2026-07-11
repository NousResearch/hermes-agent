const http = require('node:http')
const https = require('node:https')
const { Readable } = require('node:stream')

const MONEYPRINTER_GATEWAY_MEDIA_RE = /^\/api\/capabilities\/moneyprinter\/(?:download|stream)\/.+/
const MEDIA_PROTOCOL_PRIVILEGES = Object.freeze({
  corsEnabled: true,
  secure: true,
  standard: true,
  stream: true,
  supportFetchAPI: true
})

function buildGatewayMediaHeaders(sourceHeaders, token) {
  const headers = new Headers(sourceHeaders)
  headers.set('Authorization', `Bearer ${token}`)
  headers.set('X-Hermes-Session-Token', token)
  return headers
}

function decodeProtocolPath(url) {
  try {
    return decodeURIComponent(url.pathname.replace(/^\/+/, ''))
  } catch {
    throw new Error('Malformed media protocol path')
  }
}

function parseMediaProtocolUrl(rawUrl) {
  const url = new URL(String(rawUrl || ''))
  if (url.protocol !== 'hermes-media:') {
    throw new Error('Unsupported media protocol')
  }

  const decodedPath = decodeProtocolPath(url)
  if (url.hostname === 'gateway') {
    if (!MONEYPRINTER_GATEWAY_MEDIA_RE.test(decodedPath)) {
      throw new Error('Unsupported gateway media path')
    }
    return { apiPath: decodedPath, kind: 'gateway' }
  }

  if (!decodedPath) {
    throw new Error('Media file path is required')
  }
  return { filePath: decodedPath, kind: 'file' }
}

function proxyGatewayMediaRequest(targetUrl, init = {}) {
  const url = new URL(targetUrl)
  const transport = url.protocol === 'https:' ? https : http
  const method = String(init.method || 'GET').toUpperCase()
  const requestHeaders = Object.fromEntries(new Headers(init.headers).entries())

  return new Promise((resolve, reject) => {
    const upstreamRequest = transport.request(url, { headers: requestHeaders, method }, upstreamResponse => {
      const status = upstreamResponse.statusCode || 502
      const responseHeaders = new Headers()

      for (const [name, value] of Object.entries(upstreamResponse.headers)) {
        if (Array.isArray(value)) {
          for (const item of value) responseHeaders.append(name, item)
        } else if (value !== undefined) {
          responseHeaders.set(name, value)
        }
      }

      const body = method === 'HEAD' || status === 204 || status === 304 ? null : Readable.toWeb(upstreamResponse)
      resolve(
        new Response(body, {
          headers: responseHeaders,
          status,
          statusText: upstreamResponse.statusMessage || ''
        })
      )
    })

    upstreamRequest.once('error', reject)
    upstreamRequest.end()
  })
}

module.exports = {
  MEDIA_PROTOCOL_PRIVILEGES,
  buildGatewayMediaHeaders,
  parseMediaProtocolUrl,
  proxyGatewayMediaRequest
}
