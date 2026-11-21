// Whether the OS is in Do Not Disturb / Focus. Sounds the app synthesizes itself
// (the turn-end chime) bypass the OS notification centre, so the OS cannot mute
// them; ambient cues ask here first. Every probe fails open: an unreadable state
// must never silence cues the user did not ask to silence.

import { execFile } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { createClient } from 'dbus-native'

const PROBE_TIMEOUT_MS = 1000
const SERVICE = 'org.freedesktop.Notifications'

// macOS has no public Focus API; the Focus daemon writes its live assertions here.
export async function macosFocusActive(
  assertionsPath = path.join(os.homedir(), 'Library', 'DoNotDisturb', 'DB', 'Assertions.json')
): Promise<boolean> {
  try {
    const root = JSON.parse(await readFile(assertionsPath, 'utf8')) as { data?: unknown }
    const entries = Array.isArray(root?.data) ? root.data : []

    return entries.some(entry => {
      const records = (entry as { storeAssertionRecords?: unknown })?.storeAssertionRecords

      return Array.isArray(records) && records.length > 0
    })
  } catch {
    return false
  }
}

// Notification spec 1.2 `Inhibited` property (KDE Plasma, swaync, …).
async function linuxNotificationsInhibited(): Promise<boolean> {
  const bus = createClient({
    busAddress:
      process.env.DBUS_SESSION_BUS_ADDRESS ||
      `unix:path=${process.env.XDG_RUNTIME_DIR || `/run/user/${process.getuid!()}`}/bus`,
    authMethods: ['EXTERNAL'],
    direct: true,
    timeout: PROBE_TIMEOUT_MS
  })

  bus.connection.on('error', () => undefined)

  try {
    const options = { timeout: PROBE_TIMEOUT_MS }
    bus.name = await bus.invokeDbus<string>({ member: 'Hello' }, options)

    const value = await bus.invoke<unknown>(
      {
        destination: SERVICE,
        path: '/org/freedesktop/Notifications',
        interface: 'org.freedesktop.DBus.Properties',
        member: 'Get',
        signature: 'ss',
        body: [SERVICE, 'Inhibited']
      },
      options
    )

    return value === true
  } catch {
    return false
  } finally {
    bus.connection.stream.destroy()
  }
}

// GNOME Shell has no `Inhibited` property; its Do Not Disturb switch turns banners off.
function gnomeBannersOff(): Promise<boolean> {
  return new Promise(resolve => {
    execFile(
      'gsettings',
      ['get', 'org.gnome.desktop.notifications', 'show-banners'],
      { encoding: 'utf8', timeout: PROBE_TIMEOUT_MS },
      (error, stdout) => resolve(!error && stdout.trim() === 'false')
    )
  })
}

export async function isDoNotDisturbActive(platform: NodeJS.Platform = process.platform): Promise<boolean> {
  if (platform === 'darwin') {
    return macosFocusActive()
  }

  if (platform === 'linux') {
    const gnome = /gnome/i.test(process.env.XDG_CURRENT_DESKTOP ?? '')

    return (await linuxNotificationsInhibited()) || (gnome && (await gnomeBannersOff()))
  }

  return false
}
