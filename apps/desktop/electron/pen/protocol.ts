// The hermes-pen:// protocol: serves the editor bundle (blessed dir or
// Pen.app asar) with host boot params + injected chrome, self-healing on
// dead document ids.

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

import {
  lastPenThemeKind,
  PEN_AGENT_CURSOR,
  PEN_HOST_CHROME_STYLE,
  PEN_HOST_CHROME_TAGGER,
  penAgentHidden,
  penHostChrome,
  setLastPenThemeKind
} from './chrome'
import { describeDocument } from './documents'
import { ensureRuntime } from './runtime'
import { documents, events, log, runtime } from './state'

export const PEN_PROTOCOL = 'hermes-pen'

/** Handler body for protocol.handle(PEN_PROTOCOL, …). Serves the editor's
 *  static files out of Pen.app's asar; index.html gets the document's boot
 *  params injected as `window.PENCIL_INIT_PARAMS` (the same fallback global
 *  the editor reads in Pen's own VS Code webview host). */
export async function handlePenProtocolRequest(request: any, electronNet: any): Promise<any> {
  const rt = ensureRuntime()

  if (!rt) {
    return new Response('pen.dev is not installed', { status: 404 })
  }

  let url: URL

  try {
    url = new URL(request.url)
  } catch {
    return new Response('Bad request', { status: 400 })
  }

  const cleanPath = url.pathname.replace(/^\/+/, '')

  if (cleanPath === '' || cleanPath === 'index.html') {
    const docId = url.searchParams.get('doc') || ''
    const doc = documents.get(docId)

    if (!doc) {
      // A docId is only valid for the lifetime of its document in THIS
      // process — a webview reload after a close or an app restart arrives
      // here with a dead id. Self-heal instead of erroring:
      //   1. another document is live (single-canvas: there's at most one) →
      //      redirect the reload onto it;
      //   2. nothing is live → tell the renderer to drop the pane (the same
      //      close-document event every other teardown path uses) and show a
      //      quiet, theme-matched blank while it does.
      const live = [...documents.keys()][0]

      if (live) {
        return Response.redirect(`${PEN_PROTOCOL}://canvas/index.html?doc=${encodeURIComponent(live)}`, 302)
      }

      events.emit('close-document', { docId })

      return new Response(
        `<!doctype html><html><body style="margin:0;background:${penHostChrome.background}"></body></html>`,
        { headers: { 'content-type': 'text/html' } }
      )
    }

    let html = await fs.promises.readFile(path.join(rt.install.editorRoot, 'index.html'), 'utf8')

    // Seed the repaint gate with the theme this editor BOOTS with, so the
    // first host theme flip toggles it exactly once (see repaintPenTheme).
    setLastPenThemeKind(doc.device.getActiveThemeKind())

    const initParams = {
      fileURI: doc.fileURI,
      theme: lastPenThemeKind,
      connectedAgents: rt.deviceManager.getConnectedAgents(),
      isTemporary: doc.device.isTemporary(),
      // Pen's own host sends this (see ext-host.js: {fileURI, theme,
      // connectedAgents, isTemporary, isFirstLaunch, displayName,
      // hostVersion}) and it gates the first-run onboarding — the example
      // prompt chips. A hermes canvas is never pen's first launch: hermes is
      // the onboarding.
      isFirstLaunch: false,
      hostVersion: doc.device.getHostVersion(),
      displayName: describeDocument(doc).displayName
    }

    html = html.replace(
      '<script type="module"',
      [
        `<script>window.PENCIL_INIT_PARAMS = ${JSON.stringify(initParams)};</script>`,
        PEN_HOST_CHROME_STYLE,
        PEN_HOST_CHROME_TAGGER.replace('__HERMES_PEN_AGENT__', penAgentHidden ? 'hidden' : 'shown')
          .replace('__HERMES_PEN_BG__', penHostChrome.background)
          .replaceAll('__HERMES_PEN_THEME__', lastPenThemeKind || doc.device.getActiveThemeKind()),
        PEN_AGENT_CURSOR,
        '    <script type="module"'
      ].join('\n    ')
    )

    return new Response(html, { headers: { 'content-type': 'text/html; charset=utf-8' } })
  }

  const targetFile = path.join(rt.install.editorRoot, cleanPath)

  // Never escape the editor root (asar paths still normalize).
  if (!targetFile.startsWith(rt.install.editorRoot)) {
    return new Response('Forbidden', { status: 403 })
  }

  return electronNet.fetch(pathToFileURL(targetFile).toString(), {
    bypassCustomProtocolHandlers: true
  })
}

// ---------------------------------------------------------------------------
// Status + doors for the renderer
// ---------------------------------------------------------------------------

/** Canvas URL for a document — what the renderer webview loads. */
export function penCanvasUrl(docId: string): string {
  return `${PEN_PROTOCOL}://editor/index.html?doc=${encodeURIComponent(docId)}`
}

// ---------------------------------------------------------------------------
// Agent tool proxy — pen tools for the hermes agent.
//
// Rung 1: a canvas open in hermes → route through the focused document's IPC
//         (same wire the MCP server uses, minus the socket hop).
// Rung 2: no hermes canvas, but the user's Pen.app is running → dial ITS
//         socket (pencil-desktop.sock) as an MCP client. This is HUD mode:
//         hermes designs into the Pen window under the bar without ever
//         showing its own canvas.
// ---------------------------------------------------------------------------

interface PenToolResult {
  success: boolean
  result?: unknown
  error?: string
}
