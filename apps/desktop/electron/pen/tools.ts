// The agent tool surface: routes pen MCP ops to the focused canvas (or the
// user's running Pen.app), places the Hermes presence cursor, and reads
// the user's live selection.

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

import { checkpointPenDocument, revertPenDocument } from './documents'
import { documents, log, runtime } from './state'

export interface PenToolResult {
  success: boolean
  result?: unknown
  error?: string
}
import { ensureRuntime } from './runtime'

async function callFocusedCanvas(name: string, payload: Record<string, unknown>): Promise<PenToolResult | null> {
  const rt = runtime

  if (!rt || documents.size === 0) {
    return null
  }

  const { ipc } = rt.deviceManager.getFocusedResourceAndIPC()
  const target = ipc ?? [...documents.values()].find(doc => doc.ipc)?.ipc

  if (!target) {
    return null
  }

  try {
    const response: any = await target.request(name, payload)

    return {
      success: response?.success ?? true,
      result: response?.success ? response.result : undefined,
      error: response?.error
    }
  } catch (error: any) {
    return { success: false, error: error?.message || String(error) }
  }
}

/** node-ipc client for Pen's transport socket. Speaks the same framing
 *  @node-ipc/node-ipc uses: JSON messages `{type, data}` delimited by \f. */
function callPenAppSocket(name: string, payload: Record<string, unknown>, timeoutMs = 30000): Promise<PenToolResult | null> {
  const socketPath = path.join(os.homedir(), '.pencil', 'socket', 'pencil-desktop.sock')

  if (!fs.existsSync(socketPath)) {
    return Promise.resolve(null)
  }

  return new Promise(resolve => {
    const socket = net.createConnection(socketPath)
    let buffer = ''
    let clientId = ''
    const requestId = randomUUID()
    let settled = false

    const settle = (value: PenToolResult | null) => {
      if (!settled) {
        settled = true
        socket.destroy()
        resolve(value)
      }
    }

    const timer = setTimeout(() => settle({ success: false, error: 'pen.dev tool call timed out' }), timeoutMs)

    socket.on('error', () => {
      clearTimeout(timer)
      settle(null)
    })

    socket.on('data', chunk => {
      buffer += chunk.toString('utf8')

      const frames = buffer.split('\f')

      buffer = frames.pop() ?? ''

      for (const frame of frames) {
        let message: any

        try {
          message = JSON.parse(frame)
        } catch {
          continue
        }

        if (message.type !== 'tool_response') {
          continue
        }

        const data = message.data

        if (data?.request_id === 'client-id-assignment') {
          clientId = data.client_id

          socket.write(
            `${JSON.stringify({ type: 'tool_request', data: { client_id: clientId, request_id: requestId, name, payload } })}\f`
          )

          continue
        }

        if (data?.request_id === requestId) {
          clearTimeout(timer)
          settle({ success: Boolean(data.success), result: data.result, error: data.error })
        }
      }
    })
  })
}

/** The agent door: run a pen tool against whatever canvas is live. */
/** Human phrasing for the cursor label, from pen's operation names. */
const PEN_OP_LABELS: Record<string, string> = {
  copy: 'Hermes is duplicating…',
  delete: 'Hermes is deleting…',
  execute: 'Hermes is designing…',
  'export-html': 'Hermes is exporting…',
  'export-nodes': 'Hermes is exporting…',
  'get-app-state': 'Hermes is looking…',
  'get-guidelines': 'Hermes is reading…',
  'get-screenshot': 'Hermes is looking…',
  insert: 'Hermes is adding…',
  move: 'Hermes is moving…',
  replace: 'Hermes is replacing…',
  update: 'Hermes is editing…'
}

/** The live canvas WebContents, via the guest id bindPenWebview recorded.
 *  Keeps presence inside this module instead of reaching into main's drawer
 *  state — the document registry already knows who is rendering it. */
function penCanvasWebContents(): any {
  const { webContents } = require('electron')

  for (const doc of documents.values()) {
    if (!doc.guestWebContentsId) {
      continue
    }

    const target = webContents.fromId(doc.guestWebContentsId)

    if (target && !target.isDestroyed()) {
      return target
    }
  }

  return null
}

/**
 * Show the agent cursor over whatever the operation just touched, and pan the
 * viewport to it.
 *
 * Both come from the editor's own state rather than anything we track: the
 * selection after an op IS what the op affected, so its world bounds place
 * the cursor, and pen's zoom-to-selection brings it on screen. Failure is
 * silent by design — presence is a nicety and must never break the edit.
 */
async function showPenAgentCursor(name: string, follow: boolean): Promise<void> {
  const view = penCanvasWebContents()

  if (!view) {
    return
  }

  const label = PEN_OP_LABELS[name] ?? 'Hermes is working…'

  try {
    await view.executeJavaScript(
      `(() => {
        const api = window.hermesPenCursor
        if (!api) return

        // Where is the selection? The editor's scene manager owns this —
        // exposed by pen itself as __SCENE_MANAGER (IS_DEV, set at boot).
        // getWorldspaceBounds is the same call pen's rotate/zoom-to-selection
        // paths use. World coords; the cursor script maps them to screen.
        let point = null
        try {
          const sm = window.__SCENE_MANAGER
          const bounds = sm && sm.selectionManager && sm.selectionManager.getWorldspaceBounds()
          if (bounds) {
            // Bottom-right corner reads as "hand at work", not covering it.
            point = { x: bounds.x + bounds.width, y: bounds.y + bounds.height }
          }
        } catch {}

        api.place(${JSON.stringify(label)}, point)

        ${
          follow
            ? `try {
          const sm = window.__SCENE_MANAGER
          const bounds = sm && sm.selectionManager && sm.selectionManager.getWorldspaceBounds()
          if (sm && bounds && sm.camera && typeof sm.camera.ensureVisible === 'function') {
            // ensureVisible, not zoomToBounds: pans only when the work is
            // OFF-SCREEN, never yanks the zoom the user set.
            sm.camera.ensureVisible(bounds)
          }
        } catch {}`
            : ''
        }
      })()`,
      true
    )
  } catch {
    // Presence is a nicety; never let it surface as a tool failure.
  }
}

/** Park the cursor once the operation settles. */
async function idlePenAgentCursor(): Promise<void> {
  const view = penCanvasWebContents()

  if (!view) {
    return
  }

  try {
    await view.executeJavaScript(`window.hermesPenCursor && window.hermesPenCursor.idle()`, true)
  } catch {
    // Cosmetic.
  }
}

/** The user's live selection, as the agent's eyes: node ids, names, types,
 *  and world bounds, read from the scene manager pen itself exposes under
 *  IS_DEV. Selection is the deictic channel of co-design — it's how "make
 *  this blue" knows what THIS is. Empty selection is a SUCCESS with an empty
 *  list (that's an answer, not an error). */
async function readPenSelection(): Promise<PenToolResult> {
  const view = penCanvasWebContents()

  if (!view) {
    return { success: false, error: 'No pen.dev canvas is open.' }
  }

  try {
    const result = await view.executeJavaScript(
      `(() => {
        const sm = window.__SCENE_MANAGER
        if (!sm || !sm.selectionManager) return { nodes: [] }

        const nodes = []
        for (const node of sm.selectionManager.selectedNodes) {
          try {
            const entry = { id: node.id }
            if (node.name) entry.name = node.name
            if (node.type) entry.type = node.type
            const bounds =
              typeof node.getVisualWorldBounds === 'function' ? node.getVisualWorldBounds() : null
            if (bounds) {
              entry.bounds = {
                x: Math.round(bounds.x), y: Math.round(bounds.y),
                width: Math.round(bounds.width), height: Math.round(bounds.height)
              }
            }
            nodes.push(entry)
          } catch {}
        }
        return { nodes }
      })()`,
      true
    )

    return { success: true, result }
  } catch (error) {
    return { success: false, error: error instanceof Error ? error.message : String(error) }
  }
}

export async function runPenTool(name: string, payload: Record<string, unknown>): Promise<PenToolResult> {
  // Editor-side handlers are kebab-case (get-app-state); the agent tool layer
  // speaks pen's MCP names (get_app_state). Normalize once for both rungs —
  // the transport router passes names through verbatim.
  name = name.replaceAll('_', '-')

  // "What is the user pointing at?" — read the live selection off the scene
  // manager (pen's own dev door, see PEN_AGENT_CURSOR). Host-side action, not
  // an editor MCP op: it exists so "make THIS blue" resolves this/these to
  // node ids without the user having to describe what they selected.
  if (name === 'get-selection') {
    return readPenSelection()
  }

  // Canvas version control (hermes-side; pen ships none). 'revert' pops the
  // newest checkpoint — the document as it was before the agent's last edit
  // burst. Mutating ops below take the matching snapshot.
  if (name === 'revert') {
    const doc = [...documents.values()][0]

    if (!doc) {
      return { success: false, error: 'No canvas is open.' }
    }

    const stamp = await revertPenDocument(doc)

    return stamp
      ? { success: true, result: { revertedTo: new Date(stamp).toISOString() } }
      : { success: false, error: 'No checkpoint to revert to — Hermes has not edited this canvas yet.' }
  }

  // Checkpoint BEFORE the first mutating op of a burst, so "undo everything
  // Hermes just did" is one action. execute is the only door that mutates.
  if (name === 'execute') {
    const doc = [...documents.values()][0]

    if (doc) {
      await checkpointPenDocument(doc)
    }
  }

  // Presence BEFORE the op, so the cursor is already there when nodes start
  // appearing rather than catching up afterwards.
  void showPenAgentCursor(name, false)

  const fromCanvas = await callFocusedCanvas(name, payload)

  if (fromCanvas) {
    // Writes move the selection, so re-place the cursor and follow it — that's
    // the auto-pan. Reads don't move anything, so they don't yank the view.
    const write = !name.startsWith('get-') && !name.startsWith('export-')

    void showPenAgentCursor(name, write).then(idlePenAgentCursor)

    return fromCanvas
  }

  const fromPenApp = await callPenAppSocket(name, payload)

  if (fromPenApp) {
    return fromPenApp
  }

  void idlePenAgentCursor()

  return {
    success: false,
    error:
      'No pen.dev canvas is available — open a Canvas tab in Hermes, or open a document in the pen.dev desktop app.'
  }
}
