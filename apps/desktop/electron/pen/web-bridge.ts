// The pen.dev web-editor embed bridge.
//
// Speaks the hosted editor's documented embed protocol over a MessagePort —
// nothing internal, versionless, no local install. Contract: pen-embed-demo.
//
//   - main holds port1 of a MessageChannelMain; the web guest gets port2 via
//     pen-web-preload's `pen:connect` relay.
//   - editor → embedder requests are STORAGE (the embedder owns the document):
//     storage-load / storage-write / storage-{read,write,has}-asset, backed
//     by the document's .pen file + its assets beside it (assets.ts).
//   - embedder → editor requests are the MCP surface: get-mcp-schema (live
//     tool list, fetched on open / action=schema) and mcp-tool-call — plus
//     browser-import, which drops a web capture onto the canvas (web-import.ts).
//
// One canvas at a time, so one live bridge.

import fs from 'node:fs'
import path from 'node:path'

import { resolvePenAssetPath } from './assets'
import { liveDocument, penDocumentFilePath } from './documents'
import { isPenWebUrl } from './embed-url'
import { isPenSchemaAction, penToolNames, unknownPenToolError } from './mcp'
import type { PenDocument } from './state'
import { documents, events, log } from './state'
import { importedNodes, parseTopLevelNodes, type PenCanvasNode, starterFrames, topLevelNodesProbe } from './web-import-select'

const CONNECT_RETRY_MS = 1_500
const REQUEST_TIMEOUT_MS = 120_000
/** The editor gives a large page import ten minutes; a shorter wait here would call it failed while it is still working. */
const IMPORT_TIMEOUT_MS = 600_000
const READY_WAIT_MS = 30_000
const STARTER_WAIT_MS = 3_000
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
  /** Names from the last `get-mcp-schema`; the page's list, not ours. */
  toolNames: string[]
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

function assetPath(doc: PenDocument, key: string): string | null {
  const filePath = penDocumentFilePath(doc)

  return filePath ? resolvePenAssetPath(filePath, key) : null
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

/**
 * Bind once the guest origin matches. The editor's own reloads (it reloads
 * itself when a `pen:connect` reaches a page with a document already loaded)
 * are picked up by the port-close reconnect in `bindPenWebGuest`, so a guest
 * that already carries the bridge is left alone here.
 */
export function attachPenWebGuest(guestContents: any, theme: 'dark' | 'light', editorUrl: string): void {
  if (guestContents.__hermesPenWatch) {
    return
  }

  guestContents.__hermesPenWatch = true

  const onNav = () => {
    if (guestContents.isDestroyed?.()) {
      return
    }

    if (!isPenWebUrl(guestContents.getURL?.() || '', editorUrl) || bridge?.guest === guestContents) {
      return
    }

    bindPenWebGuest(guestContents, theme)
  }

  guestContents.on?.('did-navigate', onNav)
  guestContents.on?.('did-finish-load', onNav)
  onNav()
}

/**
 * Wire a web-editor guest to the live document. `pen:connect` is posted again
 * only while the page has said nothing back — a connect that reaches a page
 * with its document loaded makes the editor reload, so the retry is spaced
 * wider than the page takes to answer, and the first port the page speaks on
 * becomes the bridge's. When the page side of that port goes away (the editor
 * reloads on a document switch) the connect starts over for the new page.
 * Theme rides along at connect time only; the editor has no live theme
 * message, and a re-connect for a theme would reload it.
 */
export function bindPenWebGuest(guestContents: any, theme: 'dark' | 'light' = 'dark'): void {
  const doc = activeDoc()

  if (!doc) {
    log.warn('web guest attached with no document to bind')

    return
  }

  shutdownPenWebBridge()

  const { MessageChannelMain } = require('electron')

  const own: WebBridge = (bridge = {
    docId: doc.docId,
    guest: guestContents,
    port: null,
    ready: false,
    connectTimer: null,
    pending: new Map(),
    counter: 0,
    theme,
    toolNames: []
  })

  const current = () => bridge === own && !guestContents.isDestroyed?.()

  const stopConnecting = () => {
    if (own.connectTimer) {
      clearInterval(own.connectTimer)
      own.connectTimer = null
    }
  }

  const attempt = () => {
    if (!current()) {
      stopConnecting()

      return
    }

    const { port1, port2 } = new MessageChannelMain()

    port1.on('message', (event: any) => {
      if (!current()) {
        return
      }

      // First word from the page: this connect landed, the others never will.
      if (own.port !== port1) {
        own.port?.close()
        own.port = port1
        stopConnecting()
      }

      const message = event.data

      if (message?.kind === 'ready') {
        own.ready = true
        log.info(`pen canvas connected (${path.basename(penDocumentFilePath(doc) || doc.docId)})`)

        return
      }

      if (message?.kind === 'response') {
        const entry = own.pending.get(String(message.id))

        if (!entry) {
          return
        }

        own.pending.delete(String(message.id))
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

    // The page went away under the live port (editor reload). Anything in
    // flight is lost; the reloaded page answers a fresh connect.
    port1.on('close', () => {
      if (!current() || own.port !== port1) {
        return
      }

      own.port = null
      own.ready = false
      rejectPending('the pen editor reloaded')
      log.info('pen canvas port closed — reconnecting')
      startConnecting()
    })

    port1.start()

    guestContents.postMessage('pen-connect', { theme, fileURI: doc.fileURI }, [port2])
  }

  const startConnecting = () => {
    stopConnecting()
    attempt()
    own.connectTimer = setInterval(attempt, CONNECT_RETRY_MS)
  }

  startConnecting()
  guestContents.once?.('destroyed', () => {
    // Only the guest this bridge is bound to may take it down: the docked
    // guest dies late when the pane moves to its own window, after the new
    // guest has already bound.
    if (bridge === own) {
      shutdownPenWebBridge()
    }
  })
}

/** Re-run the connect for the live document on the bound guest (document switch). */
export function rebindPenWebGuest(): void {
  if (!bridge || bridge.guest.isDestroyed?.()) {
    return
  }

  const { guest, theme } = bridge

  bindPenWebGuest(guest, theme)
}

function rejectPending(reason: string): void {
  if (!bridge) {
    return
  }

  for (const { reject, timer } of bridge.pending.values()) {
    clearTimeout(timer)
    reject(new Error(reason))
  }

  bridge.pending.clear()
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

function bridgeRequest(method: string, payload?: unknown, timeoutMs = REQUEST_TIMEOUT_MS): Promise<unknown> {
  if (!bridge || !bridge.ready) {
    return Promise.reject(new Error('the pen canvas is not connected yet'))
  }

  const port = bridge.port
  const id = `hermes-${++bridge.counter}`

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      bridge?.pending.delete(id)
      reject(new Error(`pen request '${method}' timed out`))
    }, timeoutMs)

    bridge!.pending.set(id, { resolve, reject, timer })
    port.postMessage({ kind: 'request', id, method, payload })
  })
}

/** Live tool list from the editor. Pencil changes this; every call asks again. */
async function getPenMcpSchema(): Promise<unknown> {
  await waitForPenReady()

  let lastError: unknown

  for (let attempt = 0; attempt < SCHEMA_TRIES; attempt++) {
    try {
      const schema = await bridgeRequest('get-mcp-schema')

      if (bridge) {
        bridge.toolNames = penToolNames(schema)
      }

      return schema
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

    if (bridge?.toolNames.length === 0) {
      await getPenMcpSchema()
    }

    const unknown = unknownPenToolError(name, bridge?.toolNames ?? [])

    if (unknown) {
      return { success: false, error: unknown }
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

/** Top-level nodes as the editor sees them — an import is whatever appears that was not there before. */
async function topLevelNodes(): Promise<PenCanvasNode[]> {
  const tool = await runPenTool('execute', { input: topLevelNodesProbe })
  const text = (tool.result as McpToolResult['content'] | undefined)?.map(block => block.text ?? '').join('\n') ?? ''

  return parseTopLevelNodes(text)
}

/**
 * Drop a `PenCapturer.capture()` payload onto the canvas — pen.dev's
 * paste-from-the-web. The editor frames the inserted node itself. With `fresh`
 * (a canvas opened for this import) the editor's empty starter frame goes too.
 */
export async function importPenBrowserCapture(
  payload: string,
  { fresh = false } = {}
): Promise<{ success: boolean; nodes: PenCanvasNode[] }> {
  await waitForPenReady()

  // A just-opened document adds its starter frame a beat after the bridge is
  // ready; giving it a moment keeps the diff below honest on a slow boot.
  let before = await topLevelNodes()

  for (let waited = 0; fresh && before.length === 0 && waited < STARTER_WAIT_MS; waited += 100) {
    await sleep(100)
    before = await topLevelNodes()
  }

  const result = (await bridgeRequest('browser-import', payload, IMPORT_TIMEOUT_MS)) as { success?: boolean } | undefined

  if (result?.success !== true) {
    return { success: false, nodes: [] }
  }

  const after = await topLevelNodes()
  const starters = fresh ? starterFrames(after) : []

  if (starters.length) {
    await runPenTool('execute', { input: starters.map(node => `Delete(${JSON.stringify(node.id)})`).join('\n') })
  }

  return { success: true, nodes: importedNodes(before, after) }
}

export function shutdownPenWebBridge(): void {
  if (!bridge) {
    return
  }

  if (bridge.connectTimer) {
    clearInterval(bridge.connectTimer)
  }

  rejectPending('the pen web canvas connection was closed')

  try {
    bridge.port?.close()
  } catch {
    // already gone
  }

  bridge = null
}
