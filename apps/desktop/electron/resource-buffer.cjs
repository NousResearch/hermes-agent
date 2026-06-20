const fs = require('node:fs')
const http = require('node:http')
const https = require('node:https')
const path = require('node:path')

const { resolveReadableFileForIpc } = require('./hardening.cjs')
const { isBlockedUrl, MAX_FETCH_BYTES } = require('./url-guard.cjs')

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

async function resourceBufferFromUrl(rawUrl, options = {}) {
  if (!rawUrl) throw new Error('Missing URL')
  const url = String(rawUrl)
  const fsImpl = options.fs || fs
  const resolveFile = options.resolveReadableFileForIpc || resolveReadableFileForIpc
  const mimeForPath = options.mimeTypeForPath || mimeTypeForPath
  const blockUrl = options.isBlockedUrl || isBlockedUrl
  const maxRemoteBytes = maxBytesFromOptions(options.maxRemoteBytes)

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

  if (blockUrl(url)) throw new Error('Blocked: private URL')
  const parsed = new URL(url)
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Unsupported URL protocol: ${parsed.protocol}`)
  }
  const client = parsed.protocol === 'https:' ? https : http

  return new Promise((resolve, reject) => {
    let settled = false
    const chunks = []
    let bytesRead = 0

    const req = client.get(parsed, res => {
      const fail = error => {
        if (settled) return
        settled = true
        res.destroy()
        req.destroy(error)
        reject(error)
      }

      if ((res.statusCode || 500) >= 400) {
        settled = true
        reject(new Error(`Failed to fetch ${url}: ${res.statusCode}`))
        res.resume()
        return
      }

      const contentLength = Number(res.headers['content-length'])
      if (Number.isFinite(contentLength) && contentLength > maxRemoteBytes) {
        fail(tooLargeError(maxRemoteBytes))
        return
      }

      res.on('error', error => fail(error))
      res.on('data', chunk => {
        bytesRead += chunk.length
        if (bytesRead > maxRemoteBytes) {
          fail(tooLargeError(maxRemoteBytes))
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

module.exports = { resourceBufferFromUrl, mimeTypeForPath }
