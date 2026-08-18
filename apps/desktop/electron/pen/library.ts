// Canvas library + status: what canvases exist (~/.hermes/pens), opening
// documents, pen availability/login/icon for the renderer.

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

import { closeDocument, createLibraryDocument, describeDocument, documentIsOpen, openDocumentByUri } from './documents'
import { documents, log, type PenDocumentInfo, runtime } from './state'
import { ensureRuntime } from './runtime'

export interface PenStatus {
  available: boolean
  loggedIn: boolean
  version: string
  running: boolean
  openDocuments: PenDocumentInfo[]
  /** pen.dev's own app icon as a data URL, read at RUNTIME from the user's
   *  installed Pen.app — never bundled (their asset, upstream wins). Null
   *  when pen isn't installed; consumers fall back to a house glyph. */
  icon: null | string
}

let penIconCache: null | string = null
let penIconPending: null | Promise<null | string> = null

/** pen.dev's app icon as a data URL, decoded by macOS from the user's
 *  installed Pen.app (app.getFileIcon). Cached forever after the first hit —
 *  the icon can't change without a pen update, which relaunches us anyway. */
export function penIconDataUrl(installPath: null | string | undefined): Promise<null | string> {
  if (penIconCache || !installPath) {
    return Promise.resolve(penIconCache)
  }

  if (!penIconPending) {
    penIconPending = (async () => {
      try {
        const { app: electronApp } = require('electron')
        const image = await electronApp.getFileIcon(installPath, { size: 'normal' })

        if (image && !image.isEmpty()) {
          penIconCache = image.toDataURL()
        }
      } catch {
        // Icon is decoration; never let it break status.
      }

      return penIconCache
    })()
  }

  return penIconPending
}

export function penStatus(): PenStatus {
  const install = runtime?.install ?? findPenInstallation()

  // Kick the async prime; the resolved value rides the NEXT status call.
  // hermes:pen:status in main awaits properly, so renderers see it on the
  // first call anyway — this sync path only serves in-process callers.
  void penIconDataUrl(install?.appPath)

  return {
    available: Boolean(install),
    loggedIn: penLoggedIn(),
    version: install?.version ?? '',
    running: Boolean(runtime),
    icon: penIconCache,
    openDocuments: [...documents.values()].map(describeDocument)
  }
}

export async function openPenCanvas(options: {
  name?: string
  path?: string
  template?: string
}): Promise<PenDocumentInfo> {
  if (options.path) {
    const resolved = path.resolve(options.path)

    return openDocumentByUri(pathToFileURL(resolved).href)
  }

  return createLibraryDocument(options.template || 'pencil-new.pen', options.name)
}

// ---------------------------------------------------------------------------
// Canvas library — browse / rename / delete the user's canvases.
// ---------------------------------------------------------------------------

export interface PenLibraryItem extends PenLibraryEntry {
  /** Open in the drawer right now (so the UI can show it as active and
   *  refuse to delete it out from under itself). */
  open: boolean
  /** The live document id, when open. */
  docId: null | string
}

export function penLibrary(): { items: PenLibraryItem[]; root: string } {
  const openByPath = new Map<string, string>()

  for (const doc of documents.values()) {
    try {
      openByPath.set(path.resolve(fileURLToPath(doc.fileURI)), doc.docId)
    } catch {
      // Non-file URI — can't collide with a library path.
    }
  }

  const items = listPenLibrary().map(entry => {
    const docId = openByPath.get(path.resolve(entry.path)) ?? null

    return { ...entry, docId, open: Boolean(docId) }
  })

  return { items, root: penLibraryRoot() }
}

/** Delete a canvas. Closes the live document first — deleting the file out
 *  from under an open editor is how you get a save that resurrects it. */
export function deletePenCanvas(target: string): boolean {
  const resolved = path.resolve(target)

  for (const doc of documents.values()) {
    try {
      if (path.resolve(fileURLToPath(doc.fileURI)) === resolved) {
        closeDocument(doc.docId)
        break
      }
    } catch {
      // Not a file URI.
    }
  }

  return deletePenFromLibrary(resolved)
}

/** Rename a canvas. Refuses while it's open, for the same reason as delete:
 *  the editor holds the old path and would write it back. */
export function renamePenCanvas(target: string, nextName: string): null | string {
  const resolved = path.resolve(target)

  for (const doc of documents.values()) {
    try {
      if (path.resolve(fileURLToPath(doc.fileURI)) === resolved) {
        return null
      }
    } catch {
      // Not a file URI.
    }
  }

  return renamePenInLibrary(resolved, nextName)
}
