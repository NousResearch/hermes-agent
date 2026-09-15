import path from 'node:path'

// IPC surface for the composer's send behaviour (Settings → Keyboards): which
// keypress commits a draft, and how wide the `double-enter` window is.
//
// The renderer could hold this in localStorage like the other UI prefs, but the
// window is a number people have opinions about, so main owns it as a small JSON
// file under userData — hand-editable for values the settings control won't
// offer, and read back on every `get` so a hand-edit shows up without a rebuild.
//
// The file I/O lives in composer-send-store.ts so it can be tested without an
// Electron runtime; this module only supplies the userData path and the wiring.
import { app, ipcMain } from 'electron'

import { COMPOSER_SEND_CONFIG_FILENAME, readComposerSendPrefs, writeComposerSendPrefs } from './composer-send-store'

/** Resolved lazily: `app.setPath('userData', …)` runs during startup, so the
 *  directory is only final once main has decided on it. */
export function composerSendConfigPath(): string {
  return path.join(app.getPath('userData'), COMPOSER_SEND_CONFIG_FILENAME)
}

export function registerComposerSendIpc(): void {
  ipcMain.handle('hermes:composer-send:get', () => ({
    ...readComposerSendPrefs(composerSendConfigPath()),
    path: composerSendConfigPath()
  }))

  ipcMain.handle('hermes:composer-send:set', (_event, prefs) => ({
    ...writeComposerSendPrefs(prefs, composerSendConfigPath()),
    path: composerSendConfigPath()
  }))
}
