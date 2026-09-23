import crypto from 'node:crypto'
import fs from 'node:fs'
import http from 'node:http'
import https from 'node:https'
import path from 'node:path'

import type { resolveReadableFileForIpc } from './hardening'

interface ContentFileRuntimeDeps {
  app: { getPath: (name: 'downloads' | 'userData') => string }
  dialog: { showSaveDialog: (window: any, options: any) => Promise<{ canceled: boolean; filePath?: string }> }
  getMainWindow: () => any
  resolveReadableFileForIpc: typeof resolveReadableFileForIpc
}

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
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.webm': 'video/webm',
  '.webp': 'image/webp'
}

export function createContentFileRuntime(deps: ContentFileRuntimeDeps) {
  const { app, dialog, getMainWindow, resolveReadableFileForIpc } = deps

  function mimeTypeForPath(filePath) {
    const ext = path.extname(filePath || '').toLowerCase()

    return MEDIA_MIME_TYPES[ext] || 'application/octet-stream'
  }

  function extensionForMimeType(mimeType) {
    const type = String(mimeType || '')
      .split(';')[0]
      .trim()
      .toLowerCase()

    if (type === 'image/png') {
      return '.png'
    }

    if (type === 'image/jpeg') {
      return '.jpg'
    }

    if (type === 'image/gif') {
      return '.gif'
    }

    if (type === 'image/webp') {
      return '.webp'
    }

    if (type === 'image/bmp') {
      return '.bmp'
    }

    if (type === 'image/svg+xml') {
      return '.svg'
    }

    return ''
  }

  function filenameFromUrl(rawUrl, fallback = 'image') {
    try {
      const parsed = new URL(rawUrl)
      const base = path.basename(decodeURIComponent(parsed.pathname || ''))

      return base && base.includes('.') ? base : fallback
    } catch {
      return fallback
    }
  }

  async function resourceBufferFromUrl(rawUrl) {
    if (!rawUrl) {
      throw new Error('Missing URL')
    }

    if (rawUrl.startsWith('data:')) {
      const match = rawUrl.match(/^data:([^;,]+)?(;base64)?,(.*)$/s)

      if (!match) {
        throw new Error('Invalid data URL')
      }

      const mimeType = match[1] || 'application/octet-stream'
      const encoded = match[3] || ''
      const buffer = match[2] ? Buffer.from(encoded, 'base64') : Buffer.from(decodeURIComponent(encoded), 'utf8')

      return { buffer, mimeType }
    }

    if (/^file:/i.test(rawUrl)) {
      const { resolvedPath } = await resolveReadableFileForIpc(rawUrl, { purpose: 'Image file' })
      const buffer = await fs.promises.readFile(resolvedPath)

      return { buffer, mimeType: mimeTypeForPath(resolvedPath) }
    }

    const parsed = new URL(rawUrl)
    const client = parsed.protocol === 'https:' ? https : http

    return new Promise((resolve, reject) => {
      const req = client.get(parsed, res => {
        if ((res.statusCode || 500) >= 400) {
          reject(new Error(`Failed to fetch ${rawUrl}: ${res.statusCode}`))
          res.resume()

          return
        }

        const chunks = []
        res.on('error', reject)
        res.on('data', chunk => chunks.push(chunk))
        res.on('end', () => {
          resolve({
            buffer: Buffer.concat(chunks),
            mimeType: res.headers['content-type'] || 'application/octet-stream'
          })
        })
      })

      req.on('error', reject)
    })
  }

  async function saveImageFromUrl(rawUrl) {
    const { buffer, mimeType } = (await resourceBufferFromUrl(rawUrl)) as any
    const extension = extensionForMimeType(mimeType) || '.png'
    // Generated-image URLs (fal.media etc.) usually end in an extensionless
    // content hash. Keep the name but always guarantee an extension — without
    // one Windows saves an unopenable "All Files" blob (#image18 report).
    const baseName = filenameFromUrl(rawUrl, `image${extension}`)
    const fallbackName = path.extname(baseName) ? baseName : `${baseName}${extension}`

    let downloadsDir = ''

    try {
      downloadsDir = app.getPath('downloads')
    } catch {
      // Leave the dialog at its last-used location when the OS has no
      // Downloads directory to offer.
    }

    const result = await dialog.showSaveDialog(getMainWindow(), {
      title: 'Save Image',
      defaultPath: downloadsDir ? path.join(downloadsDir, fallbackName) : fallbackName,
      filters: [
        { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    })

    if (result.canceled || !result.filePath) {
      return false
    }

    await fs.promises.writeFile(result.filePath, buffer)

    return true
  }

  async function writeComposerImage(buffer, ext = '.png', name = '') {
    const rawExt = String(ext || '.png')
      .trim()
      .toLowerCase()

    const normalizedExt = rawExt.startsWith('.') ? rawExt : `.${rawExt}`
    const safeExt = /^\.[a-z0-9]{1,5}$/.test(normalizedExt) ? normalizedExt : '.png'
    const dir = path.join(app.getPath('userData'), 'composer-images')
    await fs.promises.mkdir(dir, { recursive: true })
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').replace('Z', '')
    const random = crypto.randomBytes(3).toString('hex')

    const baseName = String(name || '')
      .split(/[\\/]/)
      .pop()
      ?.replace(/\.[^.]+$/, '')

    const safeName = (baseName || '')
      .replace(/[^\p{L}\p{N}._-]+/gu, '_')
      .replace(/^[._-]+|[._-]+$/g, '')
      .slice(0, 80)

    const fileName = safeName ? `${safeName}_${random}${safeExt}` : `composer_${stamp}_${random}${safeExt}`
    const filePath = path.join(dir, fileName)
    await fs.promises.writeFile(filePath, buffer)

    return filePath
  }

  return { mimeTypeForPath, extensionForMimeType, resourceBufferFromUrl, saveImageFromUrl, writeComposerImage }
}
