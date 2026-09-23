import fs from 'node:fs'
import path from 'node:path'

import type { App, Clipboard, Dialog, IpcMain } from 'electron'

import { writeComposerPaste } from './composer-paste'
import {
  ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
  clampDataUrlReadMaxMb,
  DATA_URL_READ_DEFAULT_MAX_MB,
  dataUrlReadMaxBytesFromMb,
  readFileDataUrlForIpc,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  TEXT_PREVIEW_SOURCE_MAX_BYTES
} from './hardening'
import { capturePreviewContents } from './preview-capture'
import { looksBinary, PREVIEW_LANGUAGE_BY_EXT, TEXT_PREVIEW_MAX_BYTES } from './preview-target-runtime'
import { readWslWindowsClipboardImage } from './wsl-clipboard-image'
import { resolvePickerDefaultPath } from './wsl-path-bridge'

// Registers the native file, clipboard and picker doors at the original
// main-process initialization point. The window accessor remains live.
export function registerDesktopFileIpc(deps: {
  app: App
  clipboard: Clipboard
  dialog: Dialog
  electronWebContents: { fromId: (id: number) => any }
  getMainWindow: () => any
  HERMES_HOME: string
  ipcMain: IpcMain
  IS_WINDOWS: boolean
  IS_WSL: boolean
  lastContextMenuPoint: Map<number, { x: number; y: number }>
  mimeTypeForPath: (filePath: string) => string
  rememberLog: (message: string) => void
  saveGatewayFile: (payload: any) => any
  saveImageFromUrl: (url: string) => any
  writeComposerImage: (buffer: Buffer, ext?: string, name?: string) => any
}) {
  const {
    app,
    clipboard,
    dialog,
    electronWebContents,
    getMainWindow,
    HERMES_HOME,
    ipcMain,
    IS_WINDOWS,
    IS_WSL,
    lastContextMenuPoint,
    mimeTypeForPath,
    rememberLog,
    saveGatewayFile,
    saveImageFromUrl,
    writeComposerImage
  } = deps

  // Data-URL file load cap (composer attach + local previews). Main owns the
  // persisted MB value so every IPC read honours Settings → Chat without the
  // renderer having to pass maxBytes on each call. Default is 16 MB; clamp
  // lives in hardening.ts.
  const DATA_URL_READ_MAX_CONFIG_PATH = path.join(app.getPath('userData'), 'data-url-read-max.json')

  function readPersistedDataUrlReadMaxMb() {
    try {
      return clampDataUrlReadMaxMb(JSON.parse(fs.readFileSync(DATA_URL_READ_MAX_CONFIG_PATH, 'utf8')).maxMb)
    } catch {
      return DATA_URL_READ_DEFAULT_MAX_MB
    }
  }

  let dataUrlReadMaxMb = readPersistedDataUrlReadMaxMb()

  function persistDataUrlReadMaxMb(maxMb) {
    const next = clampDataUrlReadMaxMb(maxMb)
    dataUrlReadMaxMb = next

    try {
      fs.mkdirSync(path.dirname(DATA_URL_READ_MAX_CONFIG_PATH), { recursive: true })
      fs.writeFileSync(DATA_URL_READ_MAX_CONFIG_PATH, JSON.stringify({ maxMb: next }, null, 2), 'utf8')
    } catch (error) {
      rememberLog(`[data-url-read-max] write failed: ${error.message}`)
    }

    return next
  }

  ipcMain.handle('hermes:data-url-read-max:get', () => ({
    maxMb: dataUrlReadMaxMb,
    // Keep the default bytes constant visible for tests / diagnostics.
    defaultMaxMb: DATA_URL_READ_DEFAULT_MAX_MB,
    maxBytes: dataUrlReadMaxBytesFromMb(dataUrlReadMaxMb)
  }))

  ipcMain.handle('hermes:data-url-read-max:set', (_event, maxMb) => {
    const next = persistDataUrlReadMaxMb(maxMb)

    return {
      maxMb: next,
      defaultMaxMb: DATA_URL_READ_DEFAULT_MAX_MB,
      maxBytes: dataUrlReadMaxBytesFromMb(next)
    }
  })

  ipcMain.handle('hermes:readFileDataUrl', async (_event, filePath) => {
    return readFileDataUrlForIpc(filePath, {
      maxBytes: dataUrlReadMaxBytesFromMb(dataUrlReadMaxMb),
      mimeType: mimeTypeForPath(resolveRequestedPathForIpc(filePath, { purpose: 'File preview' })),
      purpose: 'File preview'
    })
  })

  // Remote attachment transfer is independent of the preview / Settings path.
  // Keep a finite cap so Electron + base64 memory stays bounded while archives
  // can exceed the default 16 MiB preview ceiling (and still fit the gateway
  // WebSocket frame limit after base64 expansion).
  ipcMain.handle('hermes:readFileDataUrlForAttach', async (_event, filePath) => {
    return readFileDataUrlForIpc(filePath, {
      maxBytes: ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
      mimeType: mimeTypeForPath(resolveRequestedPathForIpc(filePath, { purpose: 'Attachment upload' })),
      purpose: 'Attachment upload'
    })
  })

  ipcMain.handle('hermes:readFileText', async (_event, filePath) => {
    const { resolvedPath, stat } = await resolveReadableFileForIpc(filePath, {
      maxBytes: TEXT_PREVIEW_SOURCE_MAX_BYTES,
      purpose: 'Text preview'
    })

    const ext = path.extname(resolvedPath).toLowerCase()
    const handle = await fs.promises.open(resolvedPath, 'r')
    const bytesToRead = Math.min(stat.size, TEXT_PREVIEW_MAX_BYTES)

    try {
      const buffer = Buffer.alloc(bytesToRead)
      const { bytesRead } = await handle.read(buffer, 0, bytesToRead, 0)

      return {
        binary: looksBinary(buffer.subarray(0, Math.min(bytesRead, 4096))),
        byteSize: stat.size,
        language: PREVIEW_LANGUAGE_BY_EXT[ext] || 'text',
        mimeType: mimeTypeForPath(resolvedPath),
        path: resolvedPath,
        text: buffer.subarray(0, bytesRead).toString('utf8'),
        truncated: stat.size > TEXT_PREVIEW_MAX_BYTES
      }
    } finally {
      await handle.close()
    }
  })

  // Runtime desktop plugins load their FULL source through this door.
  // `hermes:readFileText` is the *preview* read and silently truncates at
  // TEXT_PREVIEW_MAX_BYTES (512 KiB) — for a plugin that means evaluating half a
  // file. Dedicated generous cap, full read, and a hard EFBIG (via maxBytes)
  // instead of truncation when the source exceeds it.
  const PLUGIN_SOURCE_MAX_BYTES = 16 * 1024 * 1024

  ipcMain.handle('hermes:readPluginSource', async (_event: unknown, filePath: unknown) => {
    const { resolvedPath, stat } = await resolveReadableFileForIpc(filePath, {
      maxBytes: PLUGIN_SOURCE_MAX_BYTES,
      purpose: 'Plugin source'
    })

    return {
      byteSize: stat.size,
      path: resolvedPath,
      text: await fs.promises.readFile(resolvedPath, 'utf8'),
      truncated: false
    }
  })

  ipcMain.handle('hermes:selectPaths', async (_event, options: any = {}) => {
    const properties = options?.directories ? ['openDirectory'] : ['openFile']

    if (options?.multiple !== false) {
      properties.push('multiSelections')
    }

    let resolvedDefaultPath

    if (options?.defaultPath) {
      try {
        // On a Windows host with a WSL backend the cwd may be a POSIX/WSL path;
        // bridge it to a UNC/drive form the native dialog can actually open.
        const bridged = IS_WINDOWS
          ? resolvePickerDefaultPath(String(options.defaultPath), undefined, options?.profile)
          : String(options.defaultPath)

        resolvedDefaultPath = bridged ? path.resolve(bridged) : undefined
      } catch {
        resolvedDefaultPath = undefined
      }
    }

    const result = await dialog.showOpenDialog(getMainWindow(), {
      title: options?.title || 'Add context',
      defaultPath: resolvedDefaultPath,
      properties: properties as any,
      filters: Array.isArray(options?.filters) ? options.filters : undefined
    })

    if (result.canceled) {
      return []
    }

    return result.filePaths
  })

  ipcMain.handle('hermes:writeClipboard', (_event, text) => {
    clipboard.writeText(String(text || ''))

    return true
  })

  // Native save-location picker (profile export etc.) — the write itself happens
  // elsewhere (the backend, for profile archives); this only picks the path.
  ipcMain.handle('hermes:selectSavePath', async (_event, options: any = {}) => {
    const result = await dialog.showSaveDialog(getMainWindow(), {
      title: options?.title || 'Save',
      defaultPath: options?.defaultPath ? String(options.defaultPath) : undefined,
      filters: Array.isArray(options?.filters) ? options.filters : undefined
    })

    if (result.canceled || !result.filePath) {
      return null
    }

    return result.filePath
  })

  // Paired reader for the GUI terminal's paste chord: the renderer's
  // navigator.clipboard.readText() throws "Document is not focused" whenever a
  // portaled overlay has focus, and there's no way to route a read through the
  // canvas. The main process has no such gate.
  ipcMain.handle('hermes:readClipboard', () => clipboard.readText())

  ipcMain.handle('hermes:saveGatewayFile', (_event, payload) => saveGatewayFile(payload))

  ipcMain.handle('hermes:saveImageFromUrl', (_event, url) => saveImageFromUrl(String(url || '')))

  // The custom context menu's edit verbs. They act on the SENDER's focused
  // element, so the renderer restores focus to the editable before invoking.
  ipcMain.handle('hermes:context-menu:edit', (event, command) => {
    const contents = event.sender

    if (command === 'copy') {
      contents.copy()
    } else if (command === 'cut') {
      contents.cut()
    } else if (command === 'paste') {
      contents.paste()
    } else if (command === 'selectAll') {
      contents.selectAll()
    }
  })

  // Copy the image under the sender's LAST context-menu gesture. Chromium only
  // exposes image bytes through copyImageAt, and only main saw the coordinates.
  ipcMain.handle('hermes:context-menu:copy-image', event => {
    const point = lastContextMenuPoint.get(event.sender.id)

    if (point) {
      event.sender.copyImageAt(point.x, point.y)
    }
  })

  ipcMain.handle('hermes:context-menu:spellcheck', (event, action) => {
    const kind = action?.kind
    const word = String(action?.word || '')

    if (!word) {
      return
    }

    if (kind === 'replace') {
      event.sender.replaceMisspelling(word)
    } else if (kind === 'add') {
      event.sender.session.addWordToSpellCheckerDictionary(word)
    }
  })

  // Guest dictionary add: the webview TAG exposes replaceMisspelling but no
  // session API, so the renderer names the guest by webContents id.
  ipcMain.handle('hermes:context-menu:guest-add-word', (_event, payload) => {
    const word = String(payload?.word || '')
    const guest = electronWebContents.fromId(Number(payload?.webContentsId))

    if (word && guest && !guest.isDestroyed()) {
      guest.session.addWordToSpellCheckerDictionary(word)
    }
  })

  ipcMain.handle('hermes:capturePreview', async (_event, payload) => {
    const guest = electronWebContents.fromId(Number(payload?.webContentsId))

    return capturePreviewContents(guest, payload?.rect, payload?.viewport)
  })

  ipcMain.handle('hermes:saveImageBuffer', async (_event, payload) => {
    const data = payload?.data

    if (!data) {
      throw new Error('saveImageBuffer: missing data')
    }

    const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data)

    return writeComposerImage(buffer, payload?.ext || '.png', payload?.name)
  })

  ipcMain.handle('hermes:savePastedText', async (_event, payload) => {
    const text = typeof payload?.text === 'string' ? payload.text : ''

    if (!text) {
      throw new Error('savePastedText: missing text')
    }

    return writeComposerPaste(HERMES_HOME, text)
  })

  ipcMain.handle('hermes:saveClipboardImage', async () => {
    const image = clipboard.readImage()

    if (image && !image.isEmpty()) {
      return writeComposerImage(image.toPNG(), '.png')
    }

    // WSL2/WSLg doesn't bridge clipboard *images* from the Windows host to the
    // Linux clipboard Electron reads, so a host screenshot looks empty above.
    // Pull it straight off the Windows clipboard via PowerShell as a fallback.
    if (IS_WSL) {
      const png = readWslWindowsClipboardImage()

      if (png) {
        return writeComposerImage(png, '.png')
      }
    }

    return ''
  })
}
