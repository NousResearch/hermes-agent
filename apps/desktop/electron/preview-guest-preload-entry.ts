// Preload for the preview pane's `<webview>` guests. main.ts installs this
// file via `will-attach-webview` on the `persist:hermes-preview` partition
// only (see `installPreviewGuestPreload`), so no other webview inherits it.
//
// The guest runs with contextIsolation, so this preload shares the guest's
// DOM but never its JavaScript world. A preview page's `target="_blank"`
// anchors (Streamlit traceback's "Ask Google" / "Ask ChatGPT" buttons —
// #112941) are intercepted here in the DOM's capture phase and handed to the
// host renderer via `sendToHost`; the host admits the scheme and routes the
// URL through the audited `hermes:openExternal` channel. A guest URL never
// becomes an Electron popup and this side never opens anything by itself.
//
// Deliberate scope: only anchor clicks are forwarded. A page's direct
// `window.open` calls stay blocked (the webview has no `allowpopups`), which
// keeps the pre-fix default for script-driven popups.

import { installGuestExternalHandoff } from './preview-guest-preload'

const electron = require('electron') as {
  ipcRenderer: { sendToHost(channel: string, ...args: unknown[]): void }
}

installGuestExternalHandoff({
  addEventListener: (type, listener, capture) => document.addEventListener(type, listener as EventListener, capture),
  sendToHost: (channel, ...args) => electron.ipcRenderer.sendToHost(channel, ...args)
})
