'use strict'

// Hidden BrowserWindow used by tier-2 link-title resolution: when curl can't
// read a page <title> (bot walls, JS-rendered pages), we briefly load the URL
// in an offscreen window and read its title. That window loads arbitrary
// user-linked pages — including YouTube/`watch` URLs that autoplay — so it must
// never be allowed to emit sound.

function linkTitleWindowOptions(partitionSession) {
  return {
    show: false,
    width: 1280,
    height: 800,
    webPreferences: {
      backgroundThrottling: false,
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
function createLinkTitleWindow(BrowserWindow, partitionSession) {
  const window = new BrowserWindow(linkTitleWindowOptions(partitionSession))

  try {
    window.webContents.setAudioMuted(true)
  } catch {
    // webContents may be unavailable in degraded/headless environments; muting
    // is best-effort and the window is destroyed within a few seconds anyway.
  }

  return window
}

module.exports = { createLinkTitleWindow, linkTitleWindowOptions }
