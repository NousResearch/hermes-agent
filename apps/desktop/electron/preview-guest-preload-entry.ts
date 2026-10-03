// Preload for the preview pane's `<webview>` guests. main.ts installs this
// file via `will-attach-webview` on the `persist:hermes-preview` partition
// only (see `installPreviewGuestPreload`), so no other webview inherits it.
//
// The guest runs with contextIsolation, so this preload shares the guest's
// DOM but never its JavaScript world. A preview page's `target="_blank"`
// anchors (Streamlit traceback's "Ask Google" / "Ask …" buttons —
// #112941) are intercepted here in the DOM's capture phase and handed to the
// host renderer via `sendToHost`; the host admits the scheme and routes the
// URL through the audited `hermes:openExternal` channel. A guest URL never
// becomes an Electron popup and this side never opens anything by itself.
//
// Only trusted anchor clicks forward URLs. A separate payload-free notification
// reports trusted pointer/key interaction so the host can select its own tab.
// A page's direct `window.open` calls stay blocked (no `allowpopups`): hooking
// them would mean reaching into the guest's JS world. Nothing is exposed there.

import { installGuestExternalHandoff, installGuestInteractionHandoff } from './preview-guest-preload'

const electron = require('electron') as {
  ipcRenderer: { sendToHost(channel: string, ...args: unknown[]): void }
}

installGuestExternalHandoff({
  addEventListener: (type, listener, capture) => document.addEventListener(type, listener, capture),
  sendToHost: (channel, ...args) => electron.ipcRenderer.sendToHost(channel, ...args)
})

installGuestInteractionHandoff({
  addEventListener: (type, listener, capture) => document.addEventListener(type, listener, capture),
  sendToHost: channel => electron.ipcRenderer.sendToHost(channel)
})
