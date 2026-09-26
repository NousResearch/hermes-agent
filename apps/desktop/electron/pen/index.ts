// Pen canvas host — Hermes embeds pen.dev's hosted editor.
//
//   - library.ts    canvas files (~/.hermes/pens) + status / open / rename
//   - documents.ts  create / open / close (one live document)
//   - sessions.ts   chat ↔ canvas ties (persist across launches)
//   - embed-url.ts  stay on /new?embed
//   - web-bridge.ts MessagePort embed protocol (storage + mcp-tool-call)
//   - web-import.ts import a preview-pane page or element onto the canvas
//   - wire.ts       ipcMain + webview attach (called once from main)
//   - state.ts      document registry + event feed

export { shutdownPenHost } from './documents'
export { syncPenWebTheme, wirePenCanvas } from './wire'
