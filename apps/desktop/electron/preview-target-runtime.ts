import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import type { App, BrowserWindow, IpcMain } from 'electron'

import { homeRelativeAttachmentCandidates, resolveReadableFileForIpc, resolveRequestedPathForIpc } from './hardening'

type PreviewTargetDependencies = {
  app: Pick<App, 'getPath'>
  directoryExists: (filePath: string) => boolean
  fileExists: (filePath: string) => boolean
  getMainWindow: () => BrowserWindow | null
  hermesHome: string
  mimeTypeForPath: (filePath: string) => string
  resolveHermesCwd: () => string
}

const PREVIEW_HTML_EXTENSIONS = new Set(['.html', '.htm'])
const PREVIEW_PDF_EXTENSIONS = new Set(['.pdf'])
const PREVIEW_WATCH_DEBOUNCE_MS = 120
const LOCAL_PREVIEW_HOSTS = new Set(['0.0.0.0', '127.0.0.1', '::1', '[::1]', 'localhost'])
export const TEXT_PREVIEW_MAX_BYTES = 512 * 1024

export const PREVIEW_LANGUAGE_BY_EXT = {
  '.c': 'c',
  '.conf': 'ini',
  '.cpp': 'cpp',
  '.css': 'css',
  '.csv': 'csv',
  '.go': 'go',
  '.graphql': 'graphql',
  '.h': 'c',
  '.hpp': 'cpp',
  '.html': 'html',
  '.java': 'java',
  '.js': 'javascript',
  '.json': 'json',
  '.jsx': 'jsx',
  '.kt': 'kotlin',
  '.lua': 'lua',
  '.md': 'markdown',
  '.mjs': 'javascript',
  '.py': 'python',
  '.rb': 'ruby',
  '.rs': 'rust',
  '.sh': 'shell',
  '.sql': 'sql',
  '.svg': 'xml',
  '.toml': 'toml',
  '.ts': 'typescript',
  '.tsx': 'tsx',
  '.txt': 'text',
  '.xml': 'xml',
  '.yaml': 'yaml',
  '.yml': 'yaml',
  '.zsh': 'shell'
}

export function looksBinary(buffer) {
  if (!buffer.length) {
    return false
  }

  let suspicious = 0

  for (const byte of buffer) {
    if (byte === 0) {
      return true
    }

    // Allow common whitespace controls: tab, LF, CR.
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      suspicious += 1
    }
  }

  return suspicious / buffer.length > 0.12
}

function previewFileMetadata(filePath, mimeType) {
  let byteSize = 0
  let binary = false

  try {
    const stat = fs.statSync(filePath)
    byteSize = stat.size

    if (!mimeType.startsWith('image/')) {
      const fd = fs.openSync(filePath, 'r')

      try {
        const sample = Buffer.alloc(Math.min(byteSize, 4096))
        const bytesRead = fs.readSync(fd, sample, 0, sample.length, 0)
        binary = looksBinary(sample.subarray(0, bytesRead))
      } finally {
        fs.closeSync(fd)
      }
    }
  } catch {
    // Metadata is best-effort; the read handlers surface hard errors later.
  }

  return {
    binary,
    byteSize,
    large: byteSize > TEXT_PREVIEW_MAX_BYTES
  }
}

export function createPreviewTargetRuntime({
  app,
  directoryExists,
  fileExists,
  getMainWindow,
  hermesHome: HERMES_HOME,
  mimeTypeForPath,
  resolveHermesCwd
}: PreviewTargetDependencies) {
  const previewWatchers = new Map<string, { close: () => void }>()

  function previewLabelForUrl(url) {
    return `${url.host}${url.pathname === '/' ? '' : url.pathname}`
  }

  function expandUserPath(filePath) {
    const value = String(filePath || '').trim()

    if (value === '~') {
      return app.getPath('home')
    }

    if (value.startsWith(`~${path.sep}`) || value.startsWith('~/')) {
      return path.join(app.getPath('home'), value.slice(2))
    }

    return value
  }

  async function previewFileTarget(rawTarget, baseDir) {
    const raw = String(rawTarget || '').trim()
    const base = baseDir ? path.resolve(expandUserPath(baseDir)) : resolveHermesCwd()

    let resolved = resolveRequestedPathForIpc(/^file:/i.test(raw) ? raw : expandUserPath(raw), {
      baseDir: base,
      purpose: 'Preview target'
    })

    // Attachment references stored in chat history are frequently HOME-relative
    // (e.g. "AppData/Local/hermes/attachments/foo.xlsx" on Windows, or
    // ".hermes/attachments/foo.xlsx" elsewhere) rather than relative to the
    // agent's working directory. The primary resolution above only tries
    // `base` (the working dir), so such a ref never exists there and the
    // preview/download 404s even though the file is present on disk (#115609).
    if (!fileExists(resolved) && !directoryExists(resolved)) {
      for (const candidate of homeRelativeAttachmentCandidates(raw, app.getPath('home'), HERMES_HOME)) {
        if (fileExists(candidate)) {
          resolved = candidate

          break
        }
      }
    }

    if (directoryExists(resolved)) {
      resolved = path.join(resolved, 'index.html')
    }

    const ext = path.extname(resolved).toLowerCase()

    if (!fileExists(resolved)) {
      return null
    }

    ;({ resolvedPath: resolved } = await resolveReadableFileForIpc(resolved, { purpose: 'Preview target' }))

    const mimeType = mimeTypeForPath(resolved)
    const metadata = previewFileMetadata(resolved, mimeType)
    const isHtml = PREVIEW_HTML_EXTENSIONS.has(ext)
    const isImage = mimeType.startsWith('image/')
    const isPdf = PREVIEW_PDF_EXTENSIONS.has(ext) || mimeType === 'application/pdf'
    const previewKind = isHtml ? 'html' : isImage ? 'image' : isPdf ? 'pdf' : metadata.binary ? 'binary' : 'text'

    return {
      binary: metadata.binary,
      byteSize: metadata.byteSize,
      kind: 'file',
      large: metadata.large,
      label: path.basename(resolved),
      language: PREVIEW_LANGUAGE_BY_EXT[ext] || 'text',
      mimeType,
      path: resolved,
      previewKind,
      source: raw,
      url: pathToFileURL(resolved).toString()
    }
  }

  function previewUrlTarget(rawTarget) {
    const raw = String(rawTarget || '').trim()
    const url = new URL(raw)

    if (!['http:', 'https:'].includes(url.protocol)) {
      return null
    }

    if (!LOCAL_PREVIEW_HOSTS.has(url.hostname.toLowerCase())) {
      return null
    }

    if (url.hostname === '0.0.0.0') {
      url.hostname = '127.0.0.1'
    }

    return {
      kind: 'url',
      label: previewLabelForUrl(url),
      source: raw,
      url: url.toString()
    }
  }

  async function normalizePreviewTarget(rawTarget, baseDir) {
    const raw = String(rawTarget || '').trim()

    if (!raw) {
      return null
    }

    try {
      if (/^https?:\/\//i.test(raw)) {
        return previewUrlTarget(raw)
      }

      return await previewFileTarget(raw, baseDir)
    } catch {
      return null
    }
  }

  async function filePathFromPreviewUrl(rawUrl) {
    const { resolvedPath } = await resolveReadableFileForIpc(String(rawUrl || ''), { purpose: 'Preview file' })

    return resolvedPath
  }

  function sendPreviewFileChanged(payload) {
    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    const { webContents } = mainWindow

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:preview-file-changed', payload)
  }

  async function watchPreviewFile(rawUrl) {
    const filePath = await filePathFromPreviewUrl(rawUrl)
    const watchDir = path.dirname(filePath)
    const targetName = path.basename(filePath)
    const id = crypto.randomBytes(12).toString('base64url')
    let timer = null

    const watcher = fs.watch(watchDir, (_eventType, filename) => {
      const changedName = filename ? path.basename(String(filename)) : ''

      if (changedName && changedName !== targetName) {
        return
      }

      if (timer) {
        clearTimeout(timer)
      }

      timer = setTimeout(() => {
        timer = null

        if (!fileExists(filePath)) {
          return
        }

        sendPreviewFileChanged({ id, path: filePath, url: pathToFileURL(filePath).toString() })
      }, PREVIEW_WATCH_DEBOUNCE_MS)
    })

    previewWatchers.set(id, {
      close: () => {
        if (timer) {
          clearTimeout(timer)
        }

        watcher.close()
      }
    })

    return { id, path: filePath }
  }

  function stopPreviewFileWatch(id) {
    const watcher = previewWatchers.get(id)

    if (!watcher) {
      return false
    }

    watcher.close()
    previewWatchers.delete(id)

    return true
  }

  function closePreviewWatchers() {
    for (const id of previewWatchers.keys()) {
      stopPreviewFileWatch(id)
    }
  }

  /** Watch a DIRECTORY for entry churn (folders appearing/vanishing) — the
   *  disk-plugin door's "new plugin folder" signal, replacing the renderer's 5s
   *  readdir poll. Same registry + change channel as the preview file watchers
   *  (the renderer reconciles on any tick; per-file edits stay on their own
   *  watches), so stopPreviewFileWatch/closePreviewWatchers manage these too. */
  function watchDirectory(rawDir) {
    const watchDir = path.resolve(String(rawDir || ''))

    if (!fs.existsSync(watchDir) || !fs.statSync(watchDir).isDirectory()) {
      throw new Error(`Not a directory: ${watchDir}`)
    }

    const id = crypto.randomBytes(12).toString('base64url')
    let timer = null

    const watcher = fs.watch(watchDir, () => {
      if (timer) {
        clearTimeout(timer)
      }

      timer = setTimeout(() => {
        timer = null
        sendPreviewFileChanged({ id, path: watchDir, url: pathToFileURL(watchDir).toString() })
      }, PREVIEW_WATCH_DEBOUNCE_MS)
    })

    previewWatchers.set(id, {
      close: () => {
        if (timer) {
          clearTimeout(timer)
        }

        watcher.close()
      }
    })

    return { id, path: watchDir }
  }

  return {
    closePreviewWatchers,
    expandUserPath,
    normalizePreviewTarget,
    stopPreviewFileWatch,
    watchDirectory,
    watchPreviewFile
  }
}

export function registerPreviewTargetIpc(
  ipcMain: Pick<IpcMain, 'handle'>,
  runtime: ReturnType<typeof createPreviewTargetRuntime>
) {
  const { normalizePreviewTarget, stopPreviewFileWatch, watchDirectory, watchPreviewFile } = runtime

  ipcMain.handle('hermes:normalizePreviewTarget', (_event, target, baseDir) =>
    normalizePreviewTarget(String(target || ''), baseDir ? String(baseDir) : '')
  )

  ipcMain.handle('hermes:watchPreviewFile', (_event, url) => watchPreviewFile(String(url || '')))

  ipcMain.handle('hermes:watchDirectory', (_event, dir) => watchDirectory(String(dir || '')))

  ipcMain.handle('hermes:stopPreviewFileWatch', (_event, id) => stopPreviewFileWatch(String(id || '')))
}
