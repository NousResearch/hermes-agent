// Canvas library + status: what canvases exist, opening documents, availability
// for the renderer. Always available — the hosted editor needs no local install.

import path from 'node:path'

import { deletePenFromLibrary, listPenLibrary, type PenLibraryEntry, penLibraryRoot, penWebEditorUrl, renamePenInLibrary } from '../pen-host'

import { closeDocument, createDocument, describeDocument, openDocument, penDocumentFilePath } from './documents'
import { documents, type PenDocumentInfo } from './state'

export interface PenStatus {
  available: boolean
  running: boolean
  openDocuments: PenDocumentInfo[]
}

export function penStatus(): PenStatus {
  return {
    available: true,
    running: documents.size > 0,
    openDocuments: [...documents.values()].map(describeDocument)
  }
}

export async function openPenCanvas(options: { name?: string; path?: string }): Promise<PenDocumentInfo> {
  if (options.path) {
    return openDocument(options.path)
  }

  return createDocument(options.name)
}

/** URL the renderer webview loads. Same hosted editor for every document;
 *  the embed bridge supplies the file via storage-load. */
export function penCanvasUrl(): string {
  return penWebEditorUrl()
}

export interface PenLibraryItem extends PenLibraryEntry {
  /** Open in the pane right now. */
  open: boolean
  /** The live document id, when open. */
  docId: null | string
}

export function penLibrary(): { items: PenLibraryItem[]; root: string } {
  const openByPath = new Map<string, string>()

  for (const doc of documents.values()) {
    const filePath = penDocumentFilePath(doc)

    if (filePath) {
      openByPath.set(path.resolve(filePath), doc.docId)
    }
  }

  const items = listPenLibrary().map(entry => {
    const docId = openByPath.get(path.resolve(entry.path)) ?? null

    return { ...entry, docId, open: Boolean(docId) }
  })

  return { items, root: penLibraryRoot() }
}

/** Delete a canvas. Closes the live document first. */
export function deletePenCanvas(target: string): boolean {
  const resolved = path.resolve(target)

  for (const doc of documents.values()) {
    const filePath = penDocumentFilePath(doc)

    if (filePath && path.resolve(filePath) === resolved) {
      closeDocument(doc.docId)
      break
    }
  }

  return deletePenFromLibrary(resolved)
}

/** Rename a canvas. Refuses while it's open. */
export function renamePenCanvas(target: string, nextName: string): null | string {
  const resolved = path.resolve(target)

  for (const doc of documents.values()) {
    const filePath = penDocumentFilePath(doc)

    if (filePath && path.resolve(filePath) === resolved) {
      return null
    }
  }

  return renamePenInLibrary(resolved, nextName)
}
