// Pen WEB-editor preload — the embed-bridge port relay.
//
// The hosted pen.dev editor (app.pen.dev/new?embed) expects the embedder to
// hand it one half of a MessageChannel by posting `pen:connect` into its main
// world. Electron can't postMessage into a <webview> guest from the outside,
// so main transfers the MessagePortMain over an internal `pen-connect` IPC and
// this preload re-posts it into the page's main world with the port attached.
//
// Mirrors pen-embed-demo/preload-editor.js. No contextBridge surface: the only
// job is forwarding the transferred port to the page.
import { ipcRenderer } from 'electron'

ipcRenderer.on('pen-connect', (event, data) => {
  window.postMessage({ type: 'pen:connect', ...(data as object) }, '*', event.ports)
})
