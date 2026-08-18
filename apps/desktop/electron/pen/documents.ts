// Document lifecycle: create/open/describe/close + autosave. Documents are
// real files (the library) from the moment they exist; closing always
// flushes; ONE live document is the invariant (closeOtherPenDocuments).

import { randomUUID } from 'node:crypto'
import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import {
  deletePenFromLibrary,
  findPenInstallation,
  listPenLibrary,
  PEN_SOCKET_APP_NAME,
  type PenInstallation,
  penLibraryPathFor,
  type PenLibraryEntry,
  penLibraryRoot,
  penLoggedIn,
  penSessionFilePath,
  penTemporaryDocumentsRoot,
  renamePenInLibrary,
  requirePenModule
} from '../pen-host'

import { createResourceDevice } from './device'
import { ensureRuntime } from './runtime'
import { documents, events, log, type PenDocument, type PenDocumentInfo, runtime } from './state'

// ---------------------------------------------------------------------------
// Document lifecycle
// ---------------------------------------------------------------------------

/**
 * A brand-new canvas, created in the LIBRARY (~/.hermes/pens/<name>/<name>.pen)
 * rather than pen's temporary-documents folder.
 *
 * Temporary documents are invisible and effectively disposable — nothing lists
 * them, and a restart strands whatever you drew. Writing a real file up front
 * means every canvas is browsable, reopenable, renameable, and deletable from
 * the moment it exists, and there's no "unsaved draft" state to lose.
 */
export async function createLibraryDocument(templateName = 'pencil-new.pen', name?: string): Promise<PenDocumentInfo> {
  const rt = ensureRuntime()

  if (!rt) {
    throw new Error('pen.dev is not installed')
  }

  if (!templateName.endsWith('.pen')) {
    templateName = `pencil-${templateName}.pen`
  }

  const templatePath = path.join(rt.install.templatesRoot, templateName)
  const newFilePath = penLibraryPathFor(name || 'Untitled')

  await fs.promises.mkdir(path.dirname(newFilePath), { recursive: true })
  await fs.promises.copyFile(templatePath, newFilePath)

  return openDocumentByUri(pathToFileURL(newFilePath).href)
}

export async function openDocumentByUri(fileURI: string): Promise<PenDocumentInfo> {
  const rt = ensureRuntime()

  if (!rt) {
    throw new Error('pen.dev is not installed')
  }

  // One tab per document — reopening an open file re-fronts its tab.
  for (const doc of documents.values()) {
    if (doc.fileURI === fileURI) {
      return describeDocument(doc)
    }
  }

  const filePath = fileURLToPath(fileURI)
  const fileContent = await fs.promises.readFile(filePath, 'utf8')
  const docId = randomUUID()

  const device = createResourceDevice(rt, fileURI, fileContent, () => {
     
    const { BrowserWindow } = require('electron')

    return BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows()[0] ?? null
  })

  const doc: PenDocument = { docId, fileURI, device, ipc: null, guestWebContentsId: null }

  documents.set(docId, doc)

  device.on('load-file', async (ev: { fileURI: string; zoomToFit?: boolean; closeCurrent?: boolean }) => {
    try {
      const info = await openDocumentByUri(ev.fileURI)

      events.emit('open-document', info)

      if (ev.closeCurrent) {
        closeDocument(docId)
        events.emit('close-document', { docId })
      }
    } catch (error) {
      log.warn('load-file failed', error)
    }
  })

  device.on('dirty-changed', (dirty: boolean) => {
    doc.ipc?.notify('dirty-changed', dirty)
    events.emit('dirty-changed', { docId, dirty })

    // AUTOSAVE. pen has no autosave of its own — its save path only runs on an
    // explicit ⌘S (userAction) or save-as, so closing the drawer, switching
    // sessions, or quitting silently discarded everything since the last
    // manual save. Hermes opens canvases on the user's behalf and reopens them
    // across restarts, so it owns this: a canvas that comes back empty is the
    // worst possible outcome.
    //
    // Debounced, because dirty-changed fires on the first edit of each burst;
    // a drag produces a continuous stream and every write is a full document
    // serialize + fsync.
    if (dirty) {
      schedulePenAutosave(doc)
    }
  })

  device.on('file-changed', (uri: string) => {
    doc.ipc?.notify('file-changed', uri)
  })

  return describeDocument(doc)
}

// ---------------------------------------------------------------------------
// Autosave
// ---------------------------------------------------------------------------

// Long enough that a drag or a burst of agent edits collapses into one write,
// short enough that nothing meaningful is ever more than a couple of seconds
// from disk.
const PEN_AUTOSAVE_DEBOUNCE_MS = 1_500

export const penAutosaveTimers = new Map<string, NodeJS.Timeout>()

/**
 * Persist a dirty document.
 *
 * Goes through the device's own `saveResource` so it takes exactly the path
 * ⌘S takes — same serializer, same dirty bookkeeping — rather than writing
 * the file behind the editor's back and leaving it thinking it's still dirty.
 *
 * `userAction: false` matters: a document that has never been saved would
 * otherwise raise a save dialog, and an autosave must never steal focus with
 * a modal. Untitled canvases don't need one anyway — hermes already created
 * them as real files in the library.
 */
export async function savePenDocument(doc: PenDocument): Promise<boolean> {
  penAutosaveTimers.delete(doc.docId)

  try {
    if (!doc.device.getIsDirty()) {
      return true
    }

    await doc.device.saveResource({ userAction: false })

    return true
  } catch (error) {
    log.warn('autosave failed', error)

    return false
  }
}

export function schedulePenAutosave(doc: PenDocument): void {
  const existing = penAutosaveTimers.get(doc.docId)

  if (existing) {
    clearTimeout(existing)
  }

  penAutosaveTimers.set(
    doc.docId,
    setTimeout(() => void savePenDocument(doc), PEN_AUTOSAVE_DEBOUNCE_MS)
  )
}

// ---------------------------------------------------------------------------
// Checkpoints — canvas version control, hermes-side (pen has none upstream)
// ---------------------------------------------------------------------------

/** Snapshots live beside the document: `<folder>/.checkpoints/<ts>.pen`.
 *  Taken at the START of an agent edit burst (first mutating op after a
 *  quiet gap), so "revert" always means "back to before Hermes touched it" —
 *  the user's own ⌘Z history inside the editor stays intact for hand edits. */
const PEN_CHECKPOINT_LIMIT = 20
const PEN_CHECKPOINT_BURST_GAP_MS = 60_000

const lastAgentEditAt = new Map<string, number>()

function checkpointDir(filePath: string): string {
  return path.join(path.dirname(filePath), '.checkpoints')
}

/** Take a checkpoint if this mutating op STARTS a burst (no agent edit in
 *  the last minute). Flushes pending autosave first so the snapshot is the
 *  document as the user last saw it, not a stale on-disk state. */
export async function checkpointPenDocument(doc: PenDocument): Promise<void> {
  const now = Date.now()
  const last = lastAgentEditAt.get(doc.docId) ?? 0

  lastAgentEditAt.set(doc.docId, now)

  if (now - last < PEN_CHECKPOINT_BURST_GAP_MS) {
    return
  }

  try {
    const filePath = doc.fileURI?.startsWith('file:') ? fileURLToPath(doc.fileURI) : null

    if (!filePath) {
      return
    }

    await savePenDocument(doc)

    if (!fs.existsSync(filePath)) {
      return
    }

    const dir = checkpointDir(filePath)

    await fs.promises.mkdir(dir, { recursive: true })
    await fs.promises.copyFile(filePath, path.join(dir, `${now}.pen`))

    // Bounded history, oldest out.
    const stamps = (await fs.promises.readdir(dir)).filter(f => f.endsWith('.pen')).sort()

    for (const stale of stamps.slice(0, Math.max(0, stamps.length - PEN_CHECKPOINT_LIMIT))) {
      await fs.promises.unlink(path.join(dir, stale)).catch(() => {})
    }
  } catch (error) {
    log.warn('checkpoint failed', error)
  }
}

/** Revert to the newest checkpoint: the document as it was before the agent's
 *  current/last burst. The checkpoint is consumed (popped), so repeated
 *  reverts walk further back through history. Returns the restored stamp or
 *  null. The document reloads through the SAME door pen's own file watching
 *  uses (load-file), so the editor repaints without a reopen. */
export async function revertPenDocument(doc: PenDocument): Promise<null | number> {
  const filePath = doc.fileURI?.startsWith('file:') ? fileURLToPath(doc.fileURI) : null

  if (!filePath) {
    return null
  }

  const dir = checkpointDir(filePath)
  const stamps = fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => f.endsWith('.pen')).sort() : []
  const latest = stamps[stamps.length - 1]

  if (!latest) {
    return null
  }

  const snapshot = path.join(dir, latest)

  // Cancel any pending autosave so it can't overwrite the revert with the
  // pre-revert buffer.
  const pending = penAutosaveTimers.get(doc.docId)

  if (pending) {
    clearTimeout(pending)
    penAutosaveTimers.delete(doc.docId)
  }

  await fs.promises.copyFile(snapshot, filePath)
  await fs.promises.unlink(snapshot).catch(() => {})
  lastAgentEditAt.delete(doc.docId)

  // Reload the editor from disk through the device's own load path.
  try {
    const content = await fs.promises.readFile(filePath, 'utf8')

    doc.device.setFileContent?.(content)
    doc.ipc?.notify('file-updated', { content, fileURI: doc.fileURI, isDirty: false, zoomToFit: false })
  } catch (error) {
    log.warn('revert reload failed', error)
  }

  return Number(latest.replace(/\.pen$/, '')) || null
}

/** Flush every dirty canvas right now — before a close, a session swap, or
 *  app quit, where waiting out the debounce would lose the tail of the work. */
export async function flushPenAutosaves(): Promise<void> {
  const pending = [...documents.values()]

  for (const timer of penAutosaveTimers.values()) {
    clearTimeout(timer)
  }

  penAutosaveTimers.clear()

  await Promise.all(pending.map(doc => savePenDocument(doc)))
}

export function describeDocument(doc: PenDocument): PenDocumentInfo {
  const isTemporary = doc.device.isTemporary()
  const basename = path.basename(doc.fileURI)

  return {
    docId: doc.docId,
    fileURI: doc.fileURI,
    displayName: isTemporary && basename === 'pencil-new.pen' ? 'Untitled' : basename.replace(/\.pen$/, ''),
    isTemporary
  }
}

/** Close every live document except `keepDocId` (pass null to close all).
 *  THE single-canvas invariant, enforced where documents live — every open
 *  path (pill, ⌘K, agent, restore, session swap) funnels through open/restore
 *  in main, and those call this. Each close autosaves first (closeDocument).
 *  Returns the docIds closed so callers can prune ties. */
export function closeOtherPenDocuments(keepDocId: null | string): string[] {
  const closed: string[] = []

  for (const docId of [...documents.keys()]) {
    if (docId !== keepDocId) {
      closeDocument(docId)
      closed.push(docId)
    }
  }

  return closed
}

/** Is this document still open in THIS launch? A temporary (never-saved)
 *  canvas exists only as a live document, so session restore uses this to
 *  tell "reattach to the draft" from "reopen the file". */
export function documentIsOpen(docId: string): boolean {
  return Boolean(docId) && documents.has(docId)
}

export function closeDocument(docId: string): void {
  const doc = documents.get(docId)

  if (!doc) {
    return
  }

  // Save BEFORE teardown. Closing the drawer, swapping sessions, and deleting
  // all route through here, and the debounce may not have fired yet — this is
  // the last moment the document still exists to be written.
  //
  // Synchronous-ish by design: kicked off before removeResource so the device
  // is still live, and awaited by flushPenAutosaves on the quit path.
  const timer = penAutosaveTimers.get(docId)

  if (timer) {
    clearTimeout(timer)
    penAutosaveTimers.delete(docId)
  }

  try {
    if (doc.device.getIsDirty()) {
      void doc.device.saveResource({ userAction: false })
    }
  } catch (error) {
    log.warn('save-on-close failed', error)
  }

  documents.delete(docId)

  try {
    runtime?.deviceManager.removeResource(doc.fileURI)
  } catch (error) {
    log.warn('removeResource failed', error)
  }
}

export function getDocument(docId: string): PenDocument | undefined {
  return documents.get(docId)
}

// ---------------------------------------------------------------------------
// Webview attach — bind the editor guest to its document's IPC host.
// ---------------------------------------------------------------------------
