// The editor's dashboard: `documents-*` requests, answered from the canvas
// library (`~/.hermes/pens/<name>/<name>.pen`). Contract: pen-embed-demo README,
// "4. Switching documents".
//
// Every library canvas is listed as a pen draft (Managed). Thumbnails are the editor's own `documents-save-preview`, kept as
// `preview.png` beside the .pen — the same file Artifacts → Pens and ⌘K show.
// Opening or creating a document is handed to the host (wire.ts), which saves
// the open canvas, ties the new one to the chat and reloads the editor.

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import { listPenLibrary, penLibraryRoot } from '../pen-host'

import { penDocumentFilePath } from './documents'
import { deletePenCanvas } from './library'
import type { PenDocument } from './state'

const DOCUMENT_KIND_MANAGED = 1
const PREVIEW_KIND_NONE = 0
const REMOVAL_DELETE = 1
const PREVIEW_FILE = 'preview.png'

export interface DashboardDocument {
  uri: string
  kind: number
  name: string
  preview: { kind: number }
  timestamps: { createdAt: number; lastEditedAt: number; lastOpenedAt?: number }
}

/** Switch the pane to a library canvas, or a new one. Set by wire.ts. */
type PenDocumentSwitcher = (target: { name?: string; path?: string }) => Promise<void>

let switcher: PenDocumentSwitcher | null = null

export function setPenDocumentSwitcher(next: PenDocumentSwitcher): void {
  switcher = next
}

/** Library canvases are pen drafts; anything outside it is a plain file. */
function inLibrary(filePath: string): boolean {
  const root = path.resolve(penLibraryRoot())

  return path.resolve(filePath).startsWith(root + path.sep)
}

function previewPathFor(filePath: string): string {
  return path.join(path.dirname(filePath), PREVIEW_FILE)
}

function uriToPath(uri: unknown): string {
  const target = fileURLToPath(String(uri))

  if (!target.endsWith('.pen')) {
    throw new Error(`not a .pen file: ${uri}`)
  }

  return target
}

export function listDashboardDocuments(live: PenDocument | null): DashboardDocument[] {
  const livePath = penDocumentFilePath(live)

  return listPenLibrary().map(entry => {
    let createdAt = entry.modifiedAt

    try {
      createdAt = fs.statSync(entry.path).birthtimeMs
    } catch {
      // keep modifiedAt
    }

    const open = livePath !== null && path.resolve(livePath) === path.resolve(entry.path)

    return {
      uri: pathToFileURL(entry.path).href,
      kind: DOCUMENT_KIND_MANAGED,
      name: entry.name,
      preview: { kind: PREVIEW_KIND_NONE },
      // Recents is the documents with a lastOpenedAt; the library has no open
      // log, so the last edit stands in for it.
      timestamps: { createdAt, lastEditedAt: entry.modifiedAt, lastOpenedAt: open ? Date.now() : entry.modifiedAt }
    }
  })
}

/** Answer one `documents-*` request; `handled: false` when `method` is not one. */
export async function handleDashboardRequest(
  doc: PenDocument,
  method: string,
  payload: any
): Promise<{ handled: boolean; value?: unknown }> {
  switch (method) {
    case 'documents-list':
      return { handled: true, value: listDashboardDocuments(doc) }
    case 'documents-previews': {
      const previews: Record<string, Uint8Array> = {}

      for (const uri of Array.isArray(payload?.uris) ? payload.uris : []) {
        try {
          previews[uri] = new Uint8Array(await fs.promises.readFile(previewPathFor(uriToPath(uri))))
        } catch {
          // no preview yet
        }
      }

      return { handled: true, value: previews }
    }

    case 'documents-save-preview': {
      const filePath = penDocumentFilePath(doc)

      if (filePath && payload?.image) {
        await fs.promises.writeFile(previewPathFor(filePath), Buffer.from(payload.image))
      }

      return { handled: true, value: undefined }
    }

    case 'documents-open':
    case 'documents-new': {
      if (!switcher) {
        throw new Error('this window cannot switch canvases')
      }

      const target: { path?: string } = method === 'documents-open' ? { path: uriToPath(payload?.uri) } : {}
      // Answer first: the switch reloads the editor, which would drop the response.
      setImmediate(() => void switcher?.(target).catch(() => undefined))

      return { handled: true, value: undefined }
    }

    case 'documents-remove': {
      const removed: string[] = []
      const deleting = payload?.removal === REMOVAL_DELETE

      for (const uri of Array.isArray(payload?.uris) ? payload.uris : []) {
        const target = uriToPath(uri)

        // Deleting only touches library canvases; removing from recents has no
        // store of its own here, so it is a no-op that reports success.
        if (deleting && (!inLibrary(target) || !deletePenCanvas(target))) {
          continue
        }

        removed.push(uri)
      }

      return { handled: true, value: removed }
    }

    default:
      return { handled: false }
  }
}
