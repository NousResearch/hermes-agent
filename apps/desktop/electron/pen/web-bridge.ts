// The pen.dev web-editor embed bridge.
//
// Speaks the hosted editor's documented embed protocol over a MessagePort —
// nothing internal, versionless, no local install. Contract: pen-embed-demo.
//
//   - main holds port1 of a MessageChannelMain; the web guest gets port2 via
//     pen-web-preload's `pen:connect` relay.
//   - editor → embedder requests are STORAGE (the embedder owns the document):
//     storage-load / storage-write / storage-{read,write,has}-asset, backed
//     by the document's .pen file + an assets/ folder beside it.
//   - embedder → editor requests are the MCP surface: get-mcp-schema (live
//     tool list, fetched on open / action=schema) and mcp-tool-call.
//
// One canvas at a time, so one live bridge.

import fs from 'node:fs'
import path from 'node:path'

import { liveDocument, penDocumentFilePath } from './documents'
import { isPenWebUrl, penEmbedDropped, restorePenEmbedUrl } from './embed-url'
import { isPenSchemaAction } from './mcp'
import type { PenDocument } from './state'
import { documents, events, log } from './state'

const CONNECT_RETRY_MS = 500
const REQUEST_TIMEOUT_MS = 120_000
const READY_WAIT_MS = 30_000
const SCHEMA_RETRY_MS = 400
const SCHEMA_TRIES = 5

interface PendingRequest {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
  timer: ReturnType<typeof setTimeout>
}

interface WebBridge {
  docId: string
  guest: any
  port: any
  ready: boolean
  connectTimer: ReturnType<typeof setInterval> | null
  pending: Map<string, PendingRequest>
  counter: number
  theme: 'dark' | 'light'
}

let bridge: WebBridge | null = null

events.on('close-document', () => {
  if (documents.size === 0) {
    shutdownPenWebBridge()
  }
})

function activeDoc(): PenDocument | null {
  return liveDocument() ?? null
}

// Assets live in an assets/ folder beside the .pen, keyed by relative path.
function assetPath(doc: PenDocument, relativePath: string): string | null {
  const filePath = penDocumentFilePath(doc)

  if (!filePath) {
    return null
  }

  const dir = path.dirname(filePath)
  const resolved = path.normalize(path.join(dir, 'assets', relativePath))

  return resolved.startsWith(dir + path.sep) ? resolved : null
}

async function handleStorageRequest(doc: PenDocument, method: string, payload: any): Promise<unknown> {
  const filePath = penDocumentFilePath(doc)

  if (!filePath) {
    throw new Error('web canvas has no backing file')
  }

  switch (method) {
    case 'storage-load': {
      const content = await fs.promises.readFile(filePath, 'utf8')
      const stat = await fs.promises.stat(filePath)

      return { filePath: path.basename(filePath), content, updatedAt: stat.mtimeMs }
    }

    case 'storage-write': {
      await fs.promises.writeFile(filePath, payload.content)

      return 0
    }

    case 'storage-read-asset': {
      const target = assetPath(doc, payload.path)

      if (!target) {
        return undefined
      }

      try {
        return new Uint8Array(await fs.promises.readFile(target))
      } catch {
        return undefined
      }
    }

    case 'storage-write-asset': {
      const target = assetPath(doc, payload.path)

      if (!target) {
        throw new Error(`invalid asset path: ${payload.path}`)
      }

      await fs.promises.mkdir(path.dirname(target), { recursive: true })
      await fs.promises.writeFile(target, Buffer.from(payload.data))

      return undefined
    }

    case 'storage-has-asset': {
      const target = assetPath(doc, payload.path)

      if (!target) {
        return false
      }

      try {
        await fs.promises.access(target)

        return true
      } catch {
        return false
      }
    }

    default:
      throw new Error(`unsupported storage request: ${method}`)
  }
}

/** Bind once the guest origin matches; bounce if Pencil strips `?embed`. */
export function attachPenWebGuest(
  guestContents: any,
  theme: 'dark' | 'light',
  editorUrl: string
): void {
  if (guestContents.__hermesPenWatch) {
    return
  }

  guestContents.__hermesPenWatch = true

  let lastBoundUrl = ''

  const onNav = () => {
    if (guestContents.isDestroyed?.()) {
      return
    }

    const url = guestContents.getURL?.() || ''

    if (penEmbedDropped(url, editorUrl)) {
      const restored = restorePenEmbedUrl(url, editorUrl)

      if (restored !== url) {
        try {
          guestContents.loadURL(restored)
        } catch {
          log.warn(`could not restore pen embed URL from ${url}`)
        }
      }

      return
    }

    if (!isPenWebUrl(url, editorUrl) || url === lastBoundUrl) {
      return
    }

    lastBoundUrl = url
    bindPenWebGuest(guestContents, theme)
  }

  guestContents.on?.('did-navigate', onNav)
  guestContents.on?.('did-navigate-in-page', onNav)
  guestContents.on?.('did-finish-load', onNav)
  onNav()
}

/** Wire a freshly-attached web-editor guest: retry `pen:connect` until ready. */
export function bindPenWebGuest(guestContents: any, theme: 'dark' | 'light' = 'dark'): void {
  const doc = activeDoc()

  if (!doc) {
    log.warn('web guest attached with no document to bind')

    return
  }

  shutdownPenWebBridge()

  const { MessageChannelMain } = require('electron')

  const attempt = () => {
    if (guestContents.isDestroyed?.()) {
      shutdownPenWebBridge()

      return
    }

    bridge?.port?.close()

    const { port1, port2 } = new MessageChannelMain()

    bridge = {
      docId: doc.docId,
      guest: guestContents,
      port: port1,
      ready: false,
      connectTimer: bridge?.connectTimer ?? null,
      pending: new Map(),
      counter: 0,
      theme
    }

    port1.on('message', (event: any) => {
      const message = event.data

      if (message?.kind === 'ready') {
        if (bridge?.connectTimer) {
          clearInterval(bridge.connectTimer)
          bridge.connectTimer = null
        }

        if (bridge) {
          bridge.ready = true
        }

        log.info(`pen canvas connected (${path.basename(penDocumentFilePath(doc) || doc.docId)})`)

        return
      }

      if (message?.kind === 'response') {
        const entry = bridge?.pending.get(String(message.id))

        if (!entry) {
          return
        }

        bridge?.pending.delete(String(message.id))
        clearTimeout(entry.timer)

        if (message.error) {
          entry.reject(new Error(`${message.error.code}: ${message.error.message}`))
        } else {
          entry.resolve(message.payload)
        }

        return
      }

      if (message?.kind === 'request') {
        void handleStorageRequest(doc, message.method, message.payload).then(
          payload => port1.postMessage({ kind: 'response', id: message.id, payload }),
          error =>
            port1.postMessage({
              kind: 'response',
              id: message.id,
              error: { code: 'ERROR', message: String(error?.message ?? error) }
            })
        )
      }
    })
    port1.start()

    guestContents.postMessage('pen-connect', { theme, fileURI: doc.fileURI }, [port2])
  }

  attempt()
  const timer = setInterval(attempt, CONNECT_RETRY_MS)

  if (bridge) {
    bridge.connectTimer = timer
  }

  guestContents.once?.('destroyed', () => shutdownPenWebBridge())
}

export function rebindPenWebGuest(theme: 'dark' | 'light'): void {
  const guest = bridge?.guest

  if (!guest || guest.isDestroyed?.()) {
    return
  }

  bindPenWebGuest(guest, theme)
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function waitForPenReady(timeoutMs = READY_WAIT_MS): Promise<void> {
  const start = Date.now()

  while (!bridge?.ready) {
    if (Date.now() - start > timeoutMs) {
      throw new Error('the pen canvas is not connected yet')
    }

    await sleep(100)
  }
}

function bridgeRequest(method: string, payload?: unknown): Promise<unknown> {
  if (!bridge || !bridge.ready) {
    return Promise.reject(new Error('the pen canvas is not connected yet'))
  }

  const port = bridge.port
  const id = `hermes-${++bridge.counter}`

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      bridge?.pending.delete(id)
      reject(new Error(`pen request '${method}' timed out`))
    }, REQUEST_TIMEOUT_MS)

    bridge!.pending.set(id, { resolve, reject, timer })
    port.postMessage({ kind: 'request', id, method, payload })
  })
}

/** Live tool list from the editor. Pencil changes this; never cache it. */
async function getPenMcpSchema(): Promise<unknown> {
  await waitForPenReady()

  let lastError: unknown

  for (let attempt = 0; attempt < SCHEMA_TRIES; attempt++) {
    try {
      return await bridgeRequest('get-mcp-schema')
    } catch (error) {
      lastError = error

      if (attempt < SCHEMA_TRIES - 1) {
        await sleep(SCHEMA_RETRY_MS)
      }
    }
  }

  throw lastError instanceof Error ? lastError : new Error(String(lastError))
}

interface McpToolResult {
  content?: Array<{ type: string; text?: string; data?: string; mimeType?: string }>
  isError?: boolean
}

export interface PenToolResult {
  success: boolean
  result?: unknown
  error?: string
}

/** Run one canvas tool. `schema` fetches the live list via `get-mcp-schema`. */
export async function runPenTool(
  name: string,
  args: Record<string, unknown>
): Promise<PenToolResult> {
  try {
    if (isPenSchemaAction(name)) {
      return { success: true, result: await getPenMcpSchema() }
    }

    const result = (await bridgeRequest('mcp-tool-call', { name, arguments: args })) as McpToolResult
    const content = result?.content ?? []

    if (result?.isError) {
      const text = content
        .filter(block => block.type === 'text')
        .map(block => block.text)
        .join('\n')

      return { success: false, error: text || 'pen tool reported a failure' }
    }

    return { success: true, result: content }
  } catch (error) {
    return { success: false, error: error instanceof Error ? error.message : String(error) }
  }
}

export function shutdownPenWebBridge(): void {
  if (!bridge) {
    return
  }

  if (bridge.connectTimer) {
    clearInterval(bridge.connectTimer)
  }

  for (const { reject, timer } of bridge.pending.values()) {
    clearTimeout(timer)
    reject(new Error('the pen web canvas connection was closed'))
  }

  try {
    bridge.port?.close()
  } catch {
    // already gone
  }

  bridge = null
}
