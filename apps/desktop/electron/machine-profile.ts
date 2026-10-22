import * as fs from 'node:fs'
import { isIP } from 'node:net'
import * as os from 'node:os'

import { app, ipcMain } from 'electron'

export interface MachineProfile {
  ageDays: number | null
  arch: string
  /** Short OS hostname for display; never persisted as a connection label. */
  hostname: string
  locale: string
  platform: NodeJS.Platform
  release: string
  username: string
}

export function registerMachineProfile(): void {
  ipcMain.handle('hermes:machine:profile', readMachineProfile)
}

function readMachineProfile(): MachineProfile {
  let ageDays: number | null = null
  let username = ''

  // Home birthtime approximates account age. Unsupported filesystems leave it unknown.
  try {
    const { birthtimeMs } = fs.statSync(os.homedir())

    if (birthtimeMs > 0) {
      ageDays = Math.max(0, Math.floor((Date.now() - birthtimeMs) / 86_400_000))
    }
  } catch {
    // Unknown age must not make a daily-use machine look newly set up.
  }

  try {
    username = os.userInfo().username
  } catch {
    // The guide can ask for a name without suggesting an OS login.
  }

  const hostname = os.hostname().trim()
  const shortHostname = hostname.split('.')[0]

  return {
    ageDays,
    arch: process.arch,
    hostname: isIP(hostname) || shortHostname.toLowerCase() === 'localhost' ? '' : shortHostname,
    locale: app.getLocale() || '',
    platform: process.platform,
    release: os.release(),
    username
  }
}
