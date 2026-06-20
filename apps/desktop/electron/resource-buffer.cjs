const fs = require('node:fs')
const http = require('node:http')
const https = require('node:https')
const path = require('node:path')

const { resolveReadableFileForIpc } = require('./hardening.cjs')
const { isBlockedUrl, resolveSafeHttpUrl, validateRedirectUrl, MAX_FETCH_BYTES } = require('./url-guard.cjs')

const MAX_REDIRECTS = 5

const MEDIA_MIME_TYPES = {
  '.avi': 'video/x-msvideo',
  '.bmp': 'image/bmp',
  '.flac': 'audio/flac',
  '.gif': 'image/gif',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.m4a': 'audio/mp4',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
  '.mp3': 'audio/mpeg',
  '.mp4': 'video/mp4',
  '.ogg': 'audio/ogg',
  '.opus': 'audio/ogg; codecs=opus',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.webm': 'video/webm',
  '.webp': 'image/webp'
}

function mimeTypeForPath(filePath) {
  const ext = path.extname(filePath || '').toLowerCase()
  return MEDIA_MIME_TYPES[ext] || 'application/octet-stream'
}

function maxBytesFromOptions(value) {
  const parsed = Number(value)
  if (Number.isFinite(parsed) && parsed > 0) return Math.floor(parsed)
  return MAX_FETCH_BYTES
}

function tooLargeError(maxBytes) {
  return new Error(`Remote resource is too large (max ${maxBytes} bytes)`)
}

function shouldBypassPrivateNetworkForUrl(options, isRedirect) {
  return Boolean(options.allowPrivateNetwork && !isRedirect)
}

async function fetchRemoteBuffer(url, options, redirectsRemaining, isRedirect = false) {
  const blockUrl = options.isBlockedUrl || isBlockedUrl
  const allowPrivateNetwork = shouldBypassPrivateNetworkForUrl(options, isRedirect)
  if (!allowPrivateNetwork && blockUrl(url)) throw new Error('Blocked: private URL')

  const safe = await resolveSafeHttpUrl(url, {
    lookup: options.lookup,
    allowPrivateNetwork
  })
  const client = safe.url.protocol === 'https:' ? https : http
  const maxRemoteBytes = maxBytesFromOptions(options.maxRemoteBytes)

  return new Promise((resolve, reject) => {
    let settled = false
    const chunks = []
    let bytesRead = 0

    const fail = (error, res, req) => {
      if (settled) return
      settled = true
      if (res) res.destroy()
      if (req) req.destroy(error)
      reject(error)
    }

    const req = client.get(safe.url, { lookup: safe.lookup }, res => {
      const statusCode = res.statusCode || 500
      const location = res.headers.location

      if (statusCode >= 300 && statusCode < 400 && location) {
        if (redirectsRemaining <= 0) {
          fail(new Error('Too many redirects'), res, req)
          return
        }

        let next
        try {
          next = validateRedirectUrl(safe.url, location).toString()
        } catch (error) {
          fail(error, res, req)
          return
        }

        settled = true
        res.resume()
        resolve(fetchRemoteBuffer(next, options, redirectsRemaining - 1, true))
        return
      }

      if (statusCode >= 400) {
        settled = true
        reject(new Error(`Failed to fetch ${url}: ${statusCode}`))
        res.resume()
        return
      }

      const contentLength = Number(res.headers['content-length'])
      if (Number.isFinite(contentLength) && contentLength > maxRemoteBytes) {
        fail(tooLargeError(maxRemoteBytes), res, req)
        return
      }

      res.on('error', error => fail(error, res, req))
      res.on('data', chunk => {
        bytesRead += chunk.length
        if (bytesRead > maxRemoteBytes) {
          fail(tooLargeError(maxRemoteBytes), res, req)
          return
        }
        chunks.push(chunk)
      })
      res.on('end', () => {
        if (settled) return
        settled = true
        resolve({
          buffer: Buffer.concat(chunks, bytesRead),
          mimeType: res.headers['content-type'] || 'application/octet-stream'
        })
      })
    })
    req.on('error', error => {
      if (settled) return
      settled = true
      reject(error)
    })
  })
}

async function resourceBufferFromUrl(rawUrl, options = {}) {
  if (!rawUrl) throw new Error('Missing URL')
  const url = String(rawUrl)
  const fsImpl = options.fs || fs
  const resolveFile = options.resolveReadableFileForIpc || resolveReadableFileForIpc
  const mimeForPath = options.mimeTypeForPath || mimeTypeForPath

  if (url.startsWith('data:')) {
    const match = url.match(/^data:([^;,]+)?(;base64)?,(.*)$/s)
    if (!match) throw new Error('Invalid data URL')
    const mimeType = match[1] || 'application/octet-stream'
    const encoded = match[3] || ''
    const buffer = match[2] ? Buffer.from(encoded, 'base64') : Buffer.from(decodeURIComponent(encoded), 'utf8')
    return { buffer, mimeType }
  }

  if (/^file:/i.test(url)) {
    const { resolvedPath } = await resolveFile(url, { purpose: 'Image file' })
    const buffer = await fsImpl.promises.readFile(resolvedPath)
    return { buffer, mimeType: mimeForPath(resolvedPath) }
  }

  const parsed = new URL(url)
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Unsupported URL protocol: ${parsed.protocol}`)
  }

  return fetchRemoteBuffer(parsed.toString(), options, options.maxRedirects ?? MAX_REDIRECTS)
}

module.exports = { resourceBufferFromUrl, mimeTypeForPath }
