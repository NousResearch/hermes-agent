import fs from 'node:fs'
import path from 'node:path'

import { BrowserWindow, ipcMain } from 'electron'

interface Options {
  preferencesPath: string
  log?: (message: string) => void
}

/**
 * Device-local download preference (#135441): when on, preview-pane downloads
 * skip the OS "Save File" dialog and land straight in Downloads. That dialog
 * is a native window no unattended agent run can answer, so a download
 * dead-ends in a `<uuid>.tmp` until a human clicks Save. Main owns the JSON
 * (same pattern as minimize-to-tray.ts); renderer windows only cache the
 * value via IPC.
 */
export function createDownloadSavePrefs(options: Options) {
  let direct = false

  const broadcast = () => {
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed()) {
        win.webContents.send('hermes:download-save-direct:changed', direct)
      }
    }
  }

  function start() {
    try {
      direct = JSON.parse(fs.readFileSync(options.preferencesPath, 'utf8')).direct === true
    } catch {
      // Missing or malformed preference preserves the save dialog default.
    }
  }

  function setDirect(on: boolean): boolean {
    direct = on === true

    try {
      fs.mkdirSync(path.dirname(options.preferencesPath), { recursive: true })
      fs.writeFileSync(`${options.preferencesPath}.tmp`, JSON.stringify({ direct }), 'utf8')
      fs.renameSync(`${options.preferencesPath}.tmp`, options.preferencesPath)
    } catch (error) {
      // The flip still applies to this session; only the relaunch
      // persistence is lost, so log instead of failing the toggle.
      options.log?.(`[download-save] write preference failed: ${(error as Error).message}`)
    }

    broadcast()

    return direct
  }

  ipcMain.handle('hermes:download-save-direct:get', () => direct)
  ipcMain.handle('hermes:download-save-direct:set', (_event, on) => setDirect(on === true))

  return { start, isEnabled: () => direct, setDirect }
}

/**
 * setSavePath overwrites an existing file without asking, while the OS dialog
 * would have offered "name (1).ext" — keep that guarantee when the
 * direct-save preference skips the dialog.
 */
export function unclaimedDownloadPath(
  downloadsDir: string,
  filename: string,
  exists: (candidate: string) => boolean = fs.existsSync
): string {
  const target = path.join(downloadsDir, filename)

  if (!exists(target)) {
    return target
  }

  const ext = path.extname(filename)
  const stem = filename.slice(0, filename.length - ext.length)

  for (let n = 1; ; n += 1) {
    const candidate = path.join(downloadsDir, `${stem} (${n})${ext}`)

    if (!exists(candidate)) {
      return candidate
    }
  }
}
