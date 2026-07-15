// Hidden BrowserWindow used by tier-2 link-title resolution: when curl can't
// read a page <title> (bot walls, JS-rendered pages), we briefly load the URL
// in an offscreen window and read its title. That window loads arbitrary
// user-linked pages, so it must never emit sound or trigger real downloads.

export function linkTitleWindowOptions(partitionSession) {
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
export function createLinkTitleWindow(BrowserWindow, partitionSession) {
  const window = new BrowserWindow(linkTitleWindowOptions(partitionSession))

  try {
    window.webContents.setAudioMuted(true)
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
