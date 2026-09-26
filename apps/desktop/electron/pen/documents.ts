// Document lifecycle: create / open / describe / close. Documents are real
// library files from the moment they exist. ONE live document is the
// invariant (closeOtherPenDocuments). Persistence is the embed bridge's
// storage-load/write against this file — there is no separate autosave.

import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import { penLibraryPathFor } from '../pen-host'

import { documents, events, type PenDocument, type PenDocumentInfo } from './state'

// Seed a brand-new canvas loads before anyone draws. Matches
// pen-embed-demo's DEFAULT_CONTENT; storage-load resolves to this on first open.
const DEFAULT_WEB_PEN = JSON.stringify({
  version: '2.6',
  children: [
    { type: 'frame', id: 'frame0', x: 0, y: 0, name: 'Frame', clip: true, width: 800, height: 600, fill: '#FFFFFF', layout: 'none' }
  ]
})

function registerDocument(filePath: string, displayName: string): PenDocumentInfo {
  const fileURI = pathToFileURL(filePath).href

  for (const doc of documents.values()) {
    if (doc.fileURI === fileURI) {
      return describeDocument(doc)
    }
  }

  const doc: PenDocument = {
    docId: randomUUID(),
    fileURI,
    displayName
  }

  documents.set(doc.docId, doc)

  return describeDocument(doc)
}

/** A brand-new canvas in `~/.hermes/pens/<name>/<name>.pen`. */
export async function createDocument(name?: string): Promise<PenDocumentInfo> {
  const displayName = (name || 'Untitled').slice(0, 60)
  const filePath = penLibraryPathFor(displayName)

  await fs.promises.mkdir(path.dirname(filePath), { recursive: true })
  await fs.promises.writeFile(filePath, DEFAULT_WEB_PEN)

  return registerDocument(filePath, displayName)
}

/** Reopen an existing library file. Re-fronts if it's already open. */
export async function openDocument(filePath: string): Promise<PenDocumentInfo> {
  const resolved = path.resolve(filePath)

  if (!fs.existsSync(resolved)) {
    throw new Error(`canvas file not found: ${resolved}`)
  }

  const displayName = path.basename(resolved).replace(/\.pen$/i, '') || 'Canvas'

  return registerDocument(resolved, displayName)
}

export function describeDocument(doc: PenDocument): PenDocumentInfo {
  return {
    docId: doc.docId,
    fileURI: doc.fileURI,
    displayName: doc.displayName || path.basename(doc.fileURI).replace(/\.pen$/i, '') || 'Canvas'
  }
}

export function liveDocument(): PenDocument | undefined {
  return [...documents.values()].at(-1)
}

export function findDocumentByPath(filePath: string): PenDocument | undefined {
  const resolved = path.resolve(filePath)

  for (const doc of documents.values()) {
    const openPath = penDocumentFilePath(doc)

    if (openPath && path.resolve(openPath) === resolved) {
      return doc
    }
  }
}

export function penDocumentFilePath(doc: { fileURI?: string } | null | undefined): null | string {
  if (!doc?.fileURI) {
    return null
  }

  try {
    return fileURLToPath(doc.fileURI)
  } catch {
    return null
  }
}

/** Close every live document except `keepDocId` (null closes all). */
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

export function documentIsOpen(docId: string): boolean {
  return Boolean(docId) && documents.has(docId)
}

export function closeDocument(docId: string): void {
  const doc = documents.get(docId)

  if (!doc) {
    return
  }

  documents.delete(docId)
  events.emit('close-document', { docId })
}

export function shutdownPenHost(): void {
  for (const docId of [...documents.keys()]) {
    closeDocument(docId)
  }
}
