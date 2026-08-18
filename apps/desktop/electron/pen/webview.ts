// <webview> guest plumbing: bind an attaching guest to its document IPC,
// and run cosmetic scripts across every live guest.

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { PEN_SOCKET_APP_NAME } from '../pen-host'

import { isPenAgentHidden } from './chrome'
import { closeDocument, createLibraryDocument } from './documents'
import { ensureRuntime } from './runtime'
import { documents, events, log, runtime } from './state'

/** Run a script in every live canvas guest (webview-hosted editor pages).
 *  Presence, chrome re-theme, agent visibility — all cosmetic, all
 *  fire-and-forget; failures never surface. */
export async function runPenGuestScript(script: string): Promise<void> {
  const { webContents } = require('electron')

  for (const doc of documents.values()) {
    if (!doc.guestWebContentsId) {
      continue
    }

    const target = webContents.fromId(doc.guestWebContentsId)

    if (target && !target.isDestroyed()) {
      await target.executeJavaScript(script, true).catch(() => {})
    }
  }
}

/** Wire a freshly attached <webview> guest showing hermes-pen://editor?doc=…
 *  to its document. Called from main.ts's did-attach-webview handler. */
export function bindPenWebview(guestContents: any): boolean {
  const url = guestContents.getURL?.() || ''

  if (!url.startsWith('hermes-pen://')) {
    return false
  }

  const rt = ensureRuntime()

  if (!rt) {
    return false
  }

  let docId = ''

  try {
    docId = new URL(url).searchParams.get('doc') || ''
  } catch {
    return false
  }

  const doc = documents.get(docId)

  if (!doc) {
    log.warn(`webview attached for unknown pen doc ${docId}`)

    return false
  }

  // Idempotent per guest: a reload rebinds, a duplicate event is a no-op.
  if (doc.ipc && doc.guestWebContentsId === guestContents.id) {
    return true
  }

   
  const { ipcMain } = require('electron')

  const penLogger = { debug: log.debug, info: log.debug, warn: log.warn, error: log.error }

  const onMessage = (callback: (message: unknown) => void) => {
    const listener = (event: any, message: unknown) => {
      if (event.sender.id === guestContents.id) {
        callback(message)
      }
    }

    ipcMain.on('ipc-message', listener)

    return () => {
      ipcMain.off('ipc-message', listener)
    }
  }

  const sendMessage = (message: unknown) => {
    if (!guestContents.isDestroyed()) {
      guestContents.send('ipc-message', message)
    }
  }

  const ipc = new rt.shared.IPCHost(onMessage, sendMessage, penLogger)

  doc.ipc = ipc
  doc.guestWebContentsId = guestContents.id

  // Editor → host: pull document content through the save round-trip (the
  // editor serializes the canvas and hands it back).
  doc.device.__setOnSave((uri: string, options?: Record<string, unknown>) =>
    ipc.request('save', { uri, ...options })
  )

  // App-level handlers Pen's PencilApp registers beside the device manager's.
  ipc.handle('get-fullscreen', () => false)
  ipc.handle('get-active-integrations', () => ({ active: [], supported: [] }))
  ipc.handle('get-mcp-config', () => {
    const config = rt.mcpLib.getMcpConfiguration({
      folderPath: rt.install.unpackedPath,
      appName: PEN_SOCKET_APP_NAME
    })

    return JSON.stringify(config)
  })
  // The editor pushes a rendered canvas preview (base64 PNG) after saves —
  // Pen.app's host writes it via its previews store; ours writes preview.png
  // beside the .pen so the library (⌘K + Artifacts) can show REAL thumbnails
  // instead of filenames. Best-effort: a failed thumbnail never surfaces.
  ipc.on('save-preview', (payload: unknown) => {
    try {
      const image =
        typeof payload === 'string'
          ? payload
          : ((payload as Record<string, unknown>)?.image as string | undefined)

      if (!image) {
        return
      }

      const filePath = doc.fileURI.startsWith('file:') ? fileURLToPath(doc.fileURI) : null

      if (!filePath) {
        return
      }

      fs.writeFileSync(path.join(path.dirname(filePath), 'preview.png'), Buffer.from(image, 'base64'))
    } catch {
      // Thumbnail is decoration.
    }
  })
  ipc.on('set-native-theme', () => {})
  ipc.on('toggle-theme', () => {})
  ipc.on('desktop-open-terminal', () => {})
  ipc.on('agent-text-size-changed', () => {})
  ipc.on('share-upload-changed', () => {})
  ipc.on('show-about', () => {})
  ipc.on('open-new-file-picker', () => {
    void createLibraryDocument('pencil-new.pen')
      .then(info => events.emit('open-document', info))
      .catch(error => log.warn('new-file-picker open failed', error))
  })
  ipc.on('add-to-chat', (message: unknown) => {
    events.emit('add-to-chat', { fileURI: doc.fileURI, message })
  })

  // The device manager wires the whole editor contract (get-session,
  // read-file, initialized → file-update, agent probes, chat sessions, …).
  rt.deviceManager.addResource(ipc, doc.device)
  rt.deviceManager.updateLastResource(doc.fileURI)

  guestContents.once('destroyed', () => {
    closeDocument(docId)
  })

  guestContents.on('focus', () => {
    rt.deviceManager.updateLastResource(doc.fileURI)
  })

  log.info(`canvas bound: ${path.basename(doc.fileURI)} (guest ${guestContents.id})`)

  return true
}

// ---------------------------------------------------------------------------
// hermes-pen:// protocol — serve the editor bundle from the installed Pen.app.
// ---------------------------------------------------------------------------
