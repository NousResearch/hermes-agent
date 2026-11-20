// Pen WEB-editor preload — the embed-bridge relay, both ways.
//
// The hosted pen.dev editor (app.pen.dev/new?embed) posts `pen:ready` to
// window.parent (this window, in a top-level <webview>) whenever its bridge
// starts listening, and expects the embedder to answer by posting `pen:connect`
// with one half of a MessageChannel. Electron can't postMessage into a guest
// from the outside, so this preload forwards `pen:ready` to main and re-posts
// main's `pen-connect` (the transferred MessagePortMain) into the page.
//
// Mirrors pen-embed-demo/preload-editor.js. No contextBridge surface.
import { ipcRenderer } from 'electron'

ipcRenderer.on('pen-connect', (event, data) => {
  window.postMessage({ type: 'pen:connect', ...(data as object) }, '*', event.ports)
})

window.addEventListener('message', event => {
  if ((event.data as { type?: unknown } | null)?.type === 'pen:ready') {
    ipcRenderer.send('pen-ready')
  }
})
