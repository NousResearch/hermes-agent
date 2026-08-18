// Pen canvas host — hermes as a pen.dev host application.
//
// The pen.dev editor is a self-contained web bundle that speaks one JSON IPC
// protocol to whatever hosts it (their Electron app, the VS Code webview, the
// headless CLI). This package makes hermes one of those hosts:
//
//   - protocol.ts  serves the editor bundle over hermes-pen:// (blessed
//                  partner bundle or the user's installed Pen.app — upstream
//                  wins, nothing vendored)
//   - device.ts    implements the host side of the editor IPC (ResourceDevice)
//   - documents.ts owns document lifecycle + autosave (library files from
//                  birth; ONE live document; closing always flushes)
//   - webview.ts   binds <webview> guests to their document IPC
//   - chrome.ts    blends the editor into hermes (theme, agent-UI hiding,
//                  injected boot assets, presence cursor markup)
//   - library.ts   the canvas library (~/.hermes/pens) + status/icon doors
//   - tools.ts     the agent tool surface (execute/get_app_state/…,
//                  presence cursor placement, live selection reads)
//   - runtime.ts   lazy bring-up/teardown of pen's transport + device manager
//   - state.ts     the shared mutable core every sibling leans on
//
// Everything degrades to "unavailable" when Pen.app is missing; nothing
// throws at import time and nothing runs until the first canvas opens.

export { isPenAgentHidden, penAgentScript, penHostChromeScript, repaintPenTheme, setPenAgentHidden, setPenHostChrome } from './chrome'
export { closeDocument, closeOtherPenDocuments, documentIsOpen } from './documents'
export { deletePenCanvas, openPenCanvas, penIconDataUrl, penLibrary, type PenLibraryItem, penStatus, type PenStatus, renamePenCanvas } from './library'
export { handlePenProtocolRequest, PEN_PROTOCOL, penCanvasUrl } from './protocol'
export { shutdownPenHost } from './runtime'
export { onPenEvent, type PenDocumentInfo } from './state'
export { type PenToolResult, runPenTool } from './tools'
export { bindPenWebview, runPenGuestScript } from './webview'
