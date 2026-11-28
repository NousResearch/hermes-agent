// Preload for the preview pane's `<webview>` guests. main.ts installs this
// file via `will-attach-webview` on the preview partition family only (see
// `installPreviewGuestPreload`), so no other webview inherits it.
//
// It deliberately does nothing today. A page's new tab (`target="_blank"`
// link or form POST, `window.open`) no longer goes through here to the OS
// browser: the guest carries `allowpopups`, main's always-deny
// `setWindowOpenHandler` forwards the request to the pane, and the pane opens
// it as a new preview tab in the same chat and cookie jar
// (electron/preview-guest-new-window.ts). The file stays as the one place a
// future guest-side bridge would go, keeping the partition-keyed install path.

export {}
