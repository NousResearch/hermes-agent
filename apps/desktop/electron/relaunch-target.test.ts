import type * as childProcess from 'node:child_process'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import { ownAppImage, relaunchNow } from './relaunch-target'

vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<typeof childProcess>()

  return { ...actual, spawn: vi.fn(actual.spawn) }
})

describe('ownAppImage', () => {
  // The runtime builds APPDIR from the raw $TMPDIR; process.execPath has its
  // symlinks resolved (TMPDIR=~/tmp on Fedora Atomic, /home -> /var/home).
  it.skipIf(process.platform === 'win32')('follows a symlinked TMPDIR to the mount it names', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-appdir-link-'))

    try {
      const mount = path.join(root, 'var-home', '.mount_HermesQ7x2Lp')
      fs.mkdirSync(mount, { recursive: true })
      fs.writeFileSync(path.join(mount, 'hermes'), '')
      fs.mkdirSync(path.join(root, 'other', '.mount_HermesQ7x2Lp'), { recursive: true })
      fs.symlinkSync(path.join(root, 'var-home'), path.join(root, 'home'))
      fs.symlinkSync(path.join(root, 'other'), path.join(root, 'elsewhere'))
      const exe = fs.realpathSync(path.join(mount, 'hermes'))
      const appImage = '/home/u/Apps/Hermes.AppImage'

      expect(ownAppImage(exe, { APPIMAGE: appImage, APPDIR: path.join(root, 'home', '.mount_HermesQ7x2Lp') })).toBe(
        appImage
      )
      expect(
        ownAppImage(exe, { APPIMAGE: appImage, APPDIR: path.join(root, 'elsewhere', '.mount_HermesQ7x2Lp') })
      ).toBeUndefined()
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
})

describe('relaunchNow', () => {
  // Chromium's relauncher closes inherited fds; a libuv spawn does not. The
  // AppImage runtime's keepalive pipe (fd 3) in the new image kept the old
  // mount and its squashfuse daemon alive for the new app's whole life; once
  // it is closed, the new image must not look anything up through that mount.
  // A spawn error arrives after the caller exited, so an image that cannot run
  // must keep this instance, its lock and its env instead.
  it.skipIf(process.platform !== 'linux')(
    'starts the image free of this mount, and stays put when it cannot',
    async () => {
      const saved = { ...process.env }
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-relaunch-fds-'))
      const image = path.join(dir, 'Hermes.AppImage')
      const report = path.join(dir, 'fds')
      const held = path.join(dir, 'held')
      const events: string[] = []

      try {
        process.env.TMPDIR = dir
        process.env.APPIMAGE = image
        process.env.APPDIR = path.dirname(process.execPath)
        delete process.env.ELECTRON_DISABLE_SANDBOX
        vi.resetModules()
        const { relaunchIfNoSandboxCameTooLate } = await import('./linux-sandbox-relaunch')

        expect(
          relaunchIfNoSandboxCameTooLate(
            {
              commandLine: { hasSwitch: name => name === 'no-sandbox' },
              relaunch: () => events.push('relaunch'),
              releaseSingleInstanceLock: () => events.push('release'),
              exit: () => events.push('exit')
            },
            () => events.push('beforeExit')
          )
        ).toBe(false)
        expect(events).toEqual([])
        expect(process.env.TMPDIR).toBe(dir)

        fs.writeFileSync(
          image,
          `#!/bin/sh\n{ echo "$PATH"; ls -l /proc/$$/fd; } > "${report}.part"\nmv "${report}.part" "${report}"\n`,
          { mode: 0o755 }
        )
        // Node marks every inherited fd close-on-exec at startup and Electron's
        // main does not, so the command relaunchNow spawns runs from a shell
        // that holds fds 3 and 9 open on a marker file, as Electron holds the
        // runtime's keepalive pipe.
        const realSpawn = (await vi.importActual<typeof childProcess>('node:child_process')).spawn
        vi.mocked(spawn).mockImplementationOnce((file, args, options) =>
          realSpawn('/bin/sh', ['-c', 'exec 3<"$0" 9<"$0"; exec "$@"', held, file, ...args], options)
        )
        fs.writeFileSync(held, '')
        // This mount's AppRun prepended it to PATH; the mount is gone soon after the child starts.
        const mount = path.join(dir, '.mount_HermesQ7x2Lp')
        relaunchNow(
          { relaunch: () => events.push('relaunch'), releaseSingleInstanceLock: () => events.push('release') },
          [],
          { kind: 'appimage', image },
          { ...process.env, APPDIR: mount, PATH: `${mount}:${mount}/usr/sbin:/usr/bin:/bin` }
        )

        await vi.waitFor(() => expect(fs.existsSync(report)).toBe(true), { timeout: 5000 })
        expect(events).toEqual(['release'])
        const [childPath, ...fds] = fs.readFileSync(report, 'utf8').split('\n')
        expect(childPath).toBe('/usr/bin:/bin')
        expect(fds.join('\n')).not.toContain(held)
      } finally {
        for (const key of Object.keys(process.env)) {
          if (!(key in saved)) {
            delete process.env[key]
          }
        }

        Object.assign(process.env, saved)
        fs.rmSync(dir, { recursive: true, force: true })
      }
    }
  )
})
