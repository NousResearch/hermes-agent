import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { ownAppImage } from './relaunch-target'

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
