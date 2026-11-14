// Hidden BrowserWindow used by tier-2 link-title resolution: when curl can't
// read a page <title> (bot walls, JS-rendered pages), we briefly load the URL
// in an offscreen window and read its title. That window loads arbitrary
// user-linked pages, so it must never emit sound, trigger real downloads, or
// become visible — on Windows Desktop a hidden title-fetch window has still
// painted a blank top-level frame over the chat (#64867), so the invisibility
// contract is encoded in the launch options, not just show: false.

import { createWindowOpenHandler } from './window-open-policy'

export function linkTitleWindowOptions(partitionSession) {
  return {
    show: false,
    width: 1280,
    height: 800,
    // Park far off-screen and keep it off the taskbar so a show:false leak can
    // never cover the chat window or steal focus (#64867).
    x: -20000,
    y: -20000,
    skipTaskbar: true,
    focusable: false,
    webPreferences: {
      // Deliberately throttled: this hidden window loads arbitrary user-linked
      // pages, and an unthrottled heavy page burns full CPU for the window's
      // whole lifetime. Title resolution rides load events
      // (page-title-updated / did-finish-load) plus main-process timers, none
      // of which the renderer clamp touches — hidden-page throttling only
      // slows the page's own timer-driven JS, and the grace window already
      // absorbs that.
      contextIsolation: true,
      javascript: true,
      nodeIntegration: false,
      sandbox: true,
      session: partitionSession,
      webSecurity: true
    }
  }
}

// Create the offscreen title-fetch window and immediately mute it. Without the
// mute, autoplaying media on the loaded page (e.g. a YouTube link) leaks ~2s of
// audio every time a session containing such links is re-rendered. See #49505.
export function createLinkTitleWindow(BrowserWindow, partitionSession) {
  const window = new BrowserWindow(linkTitleWindowOptions(partitionSession))

  try {
    // Belt-and-suspenders with the off-screen show:false options (#64867):
    // a zero-opacity hidden window cannot flash a blank frame even if the
    // platform paints it.
    window.setOpacity?.(0)
    window.hide?.()
  } catch {
    // best-effort; title fetch still works without hide/opacity
  }

  try {
    window.webContents.setAudioMuted(true)
    // Loads arbitrary user-linked pages on render; it only needs the title, so
    // a popup from that page never has a reason to exist (GHSA-9f4c-93c8-jc8g).
    window.webContents.setWindowOpenHandler(createWindowOpenHandler())
  } catch {
    // webContents may be unavailable in degraded/headless environments; muting
    // is best-effort and the window is destroyed within a few seconds anyway.
  }

  return window
}

// Cancel any download the title-fetch window triggers. Without this, a link
// artifact URL served with Content-Disposition: attachment auto-downloads every
// time the Artifacts page renders and fetchLinkTitle loads it.
export function guardLinkTitleSession(partitionSession) {
  try {
    partitionSession.on('will-download', (_event, item) => item.cancel())
  } catch {
    // best-effort; worst case is a spurious download
  }
}

// Read the page title from a title-fetch window. Callers schedule this from
// timers that can fire after finish() destroys the window, so every access must
// guard isDestroyed and swallow Electron's "Object has been destroyed" throws.
export function readLinkTitleWindowTitle(window) {
  try {
    if (!window || window.isDestroyed()) {
      return ''
    }

    const contents = window.webContents

    if (!contents || contents.isDestroyed()) {
      return ''
    }

    return contents.getTitle() || ''
  } catch {
    return ''
  }
}
