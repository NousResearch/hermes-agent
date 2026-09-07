// Electron wiring for the pen canvas: webview attach + ipcMain doors.
// Called once from main. Session ties persist next to desktop userData.

import fs from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, ipcMain, nativeTheme, shell } from 'electron'

import { penWebEditorUrl } from '../pen-host'

import {
  closeOtherPenDocuments,
  describeDocument,
  documentIsOpen,
  findDocumentByPath,
  liveDocument,
  penDocumentFilePath
} from './documents'
import { isPenWebUrl } from './embed-url'
import { deletePenCanvas, openPenCanvas, penCanvasUrl, penLibrary, penStatus, renamePenCanvas } from './library'
import {
  forgetPenSession,
  penPaneAction,
  readPenSessions,
  rememberPenSession,
  retargetPenSessionPaths,
  samePenPath,
  sessionIdByCanvasPath
} from './sessions'
import { onPenEvent } from './state'
import { attachPenWebGuest, rebindPenWebGuest, runPenTool } from './web-bridge'

const penDocSessions = new Map<string, string>()

function sessionsFile(): string {
  return path.join(app.getPath('userData'), 'pen-canvas-sessions.json')
}

export function penWebTheme(): 'dark' | 'light' {
  return nativeTheme.shouldUseDarkColors ? 'dark' : 'light'
}

export function syncPenWebTheme(): void {
  rebindPenWebGuest(penWebTheme())
}

function broadcastPenEvent(event: string, payload: unknown): void {
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) {
      win.webContents.send('hermes:pen:event', { event, payload })
    }
  }
}

function wirePenWebviewGuests(opts: { preloadPath: string; windowBackground: () => string }): void {
  app.on('web-contents-created', (_event, contents) => {
    contents.on('will-attach-webview', (_e, webPreferences, params) => {
      if (!isPenWebUrl(String(params.src || ''), penWebEditorUrl())) {
        return
      }

      webPreferences.preload = opts.preloadPath
      webPreferences.contextIsolation = true
      webPreferences.nodeIntegration = false
      webPreferences.sandbox = false
    })

    contents.on('did-attach-webview', (_e, guest) => {
      attachPenWebGuest(guest, penWebTheme(), penWebEditorUrl())

      try {
        guest.setBackgroundColor(opts.windowBackground())
      } catch {
        // Cosmetic.
      }
    })
  })
}

function wirePenIpc(): void {
  const store = sessionsFile()

  for (const event of ['open-document', 'close-document']) {
    onPenEvent(event, payload => broadcastPenEvent(event, payload))
  }

  onPenEvent('open-document', payload => {
    const sessionId = payload?.docId ? penDocSessions.get(payload.docId) : null

    if (sessionId) {
      rememberPenSession(store, sessionId, { docId: payload.docId, path: penDocumentFilePath(payload), closed: false })
    }
  })

  onPenEvent('close-document', payload => {
    if (payload?.docId) {
      penDocSessions.delete(payload.docId)
    }
  })

  const front = (doc: { docId: string; fileURI?: string }, sessionId: string | undefined, rebind: boolean) => {
    for (const closedId of closeOtherPenDocuments(doc.docId)) {
      penDocSessions.delete(closedId)
    }

    if (sessionId) {
      penDocSessions.set(doc.docId, sessionId)
    }

    rememberPenSession(store, sessionId, {
      docId: doc.docId,
      path: penDocumentFilePath(doc),
      closed: false
    })

    if (rebind) {
      rebindPenWebGuest(penWebTheme())
    }

    return { doc, url: penCanvasUrl() }
  }

  ipcMain.handle('hermes:pen:status', async () => penStatus())

  ipcMain.handle('hermes:pen:open', async (_event, options) => {
    const { sessionId, ...openOptions } = options || {}
    const prior = penDocumentFilePath(liveDocument())
    const doc = await openPenCanvas(openOptions)

    return front(doc, sessionId, !samePenPath(prior, penDocumentFilePath(doc)))
  })

  // Draft chats open a canvas before they have a session id. Adopt ties it
  // once the chat is promoted so restore/reopen still work.
  ipcMain.handle('hermes:pen:adopt', (_event, sessionId) => {
    if (!sessionId) {
      return false
    }

    const openDocs = penStatus().openDocuments
    const tied = [...penDocSessions.keys()]
    const doc = (tied.length > 0 ? openDocs.find(d => d.docId === tied[0]) : openDocs[0]) ?? openDocs[0]

    if (!doc) {
      return false
    }

    penDocSessions.set(doc.docId, sessionId)
    rememberPenSession(store, sessionId, {
      docId: doc.docId,
      path: penDocumentFilePath(doc),
      closed: false
    })

    return true
  })

  ipcMain.handle('hermes:pen:session', (_event, sessionId) => {
    const entry = sessionId ? readPenSessions(store)[sessionId] ?? null : null

    if (!entry) {
      return null
    }

    const restorable = Boolean(entry.path) || documentIsOpen(entry.docId ?? '')

    return restorable ? entry : null
  })

  ipcMain.handle('hermes:pen:restore', async (_event, sessionId) => {
    const entry = sessionId ? readPenSessions(store)[sessionId] ?? null : null

    if (!entry) {
      return null
    }

    const live = liveDocument()
    const byPath = entry.path ? findDocumentByPath(entry.path) : undefined
    const action = penPaneAction(penDocumentFilePath(live), entry)
    const keep = byPath ?? live

    if (action === 'keep' && keep) {
      return front(describeDocument(keep), sessionId, false)
    }

    if (action !== 'show' || !entry.path || !fs.existsSync(entry.path)) {
      if (!entry.path || !fs.existsSync(entry.path)) {
        forgetPenSession(store, sessionId)
      }

      return null
    }

    const doc = await openPenCanvas({ path: entry.path })

    return front(doc, sessionId, true)
  })

  ipcMain.handle('hermes:pen:library', () => {
    const library = penLibrary()
    const sessionByPath = sessionIdByCanvasPath(readPenSessions(store))

    return {
      ...library,
      items: library.items.map(item => ({
        ...item,
        sessionId: sessionByPath.get(path.resolve(item.path)) ?? null
      }))
    }
  })

  ipcMain.handle('hermes:pen:library-delete', (_event, target) => deletePenCanvas(String(target || '')))

  ipcMain.handle('hermes:pen:library-rename', (_event, target, nextName) => {
    const oldResolved = path.resolve(String(target || ''))
    const renamed = renamePenCanvas(String(target || ''), String(nextName || ''))

    if (renamed) {
      retargetPenSessionPaths(store, oldResolved, renamed)
    }

    return renamed
  })

  ipcMain.handle('hermes:pen:reveal', (_event, target) => {
    const file = String(target || '')

    if (file) {
      shell.showItemInFolder(file)
    }
  })

  ipcMain.handle('hermes:pen:close', (_event, options) => {
    // ✕ puts the canvas away; the tie stays so the reopen pill can bring it back.
    if (!options?.keep) {
      for (const sessionId of penDocSessions.values()) {
        rememberPenSession(store, sessionId, { closed: true })
      }
    }

    penDocSessions.clear()
    closeOtherPenDocuments(null)
  })

  ipcMain.handle('hermes:pen:tool', async (_event, name, payload) =>
    runPenTool(String(name || ''), payload && typeof payload === 'object' ? payload : {})
  )
}

export function wirePenCanvas(opts: { preloadPath: string; windowBackground: () => string }): void {
  wirePenWebviewGuests(opts)
  wirePenIpc()
}
