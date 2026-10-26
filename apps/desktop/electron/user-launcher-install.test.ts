import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import { resolveSourceInstallationBackend } from './source-backend'
import { userLauncherInstallRoot } from './updater-process'

afterEach((): void => {
  vi.unstubAllEnvs()
})

function sourceTree(root: string): void {
  fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
  fs.mkdirSync(path.join(root, 'pm'))
  fs.writeFileSync(path.join(root, 'hermes_cli', 'main.py'), '')
  fs.writeFileSync(path.join(root, 'hermes_cli', '_launchers.py'), '')
}

function publishLauncher(dir: string, reported: string): string {
  fs.mkdirSync(dir, { recursive: true })
  const launcher: string = path.join(dir, process.platform === 'win32' ? 'hermes.cmd' : 'hermes')

  const body: string =
    process.platform === 'win32'
      ? `@echo off\r\necho Install directory: ${reported}\r\n`
      : `#!/bin/sh\nprintf '%s\\n' 'Install directory: ${reported}'\n`

  fs.writeFileSync(launcher, body, { mode: 0o755 })

  return launcher
}

// The reported machine: setup-hermes.sh published ~/.local/bin/hermes for a
// clone outside ~/.hermes/hermes-agent, and a Finder/Dock launch inherited
// launchd's PATH, which has no ~/.local/bin. Desktop must still find it.
test.skipIf(process.platform === 'win32')(
  'a ~/.local/bin launcher resolves its non-canonical install on a GUI PATH without ~/.local/bin',
  async (): Promise<void> => {
    const base: string = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'user-launcher-')))

    try {
      const home: string = path.join(base, 'home')
      const root: string = path.join(home, 'src', 'hermes-agent')
      sourceTree(root)
      const launcher: string = publishLauncher(path.join(home, '.local', 'bin'), root)
      vi.stubEnv('HOME', home)
      vi.stubEnv('PATH', '/usr/bin:/bin:/usr/sbin:/sbin')

      const found = userLauncherInstallRoot(false, path.join(home, '.hermes'))
      assert.deepEqual(found, { launcher, root })

      const backend = await resolveSourceInstallationBackend(root, ['serve'], {
        isWindows: false,
        hermesHome: path.join(home, '.hermes')
      })

      assert.equal(backend?.command, launcher)
      assert.equal(backend?.root, root)
      assert.deepEqual(backend?.args, ['serve'])
    } finally {
      fs.rmSync(base, { recursive: true, force: true })
    }
  }
)

test('a user-bin launcher is ignored when it reports no Hermes source tree', (): void => {
  const base: string = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'user-launcher-')))

  try {
    const home: string = path.join(base, 'home')
    const hermesHome: string = path.join(home, '.hermes')
    const notSource: string = path.join(base, 'elsewhere')
    fs.mkdirSync(notSource)
    publishLauncher(path.join(hermesHome, 'bin'), notSource)
    vi.stubEnv('HOME', home)
    vi.stubEnv('USERPROFILE', home)
    vi.stubEnv('LOCALAPPDATA', path.join(home, 'AppData', 'Local'))

    assert.equal(userLauncherInstallRoot(process.platform === 'win32', hermesHome), null)

    sourceTree(notSource)
    assert.equal(userLauncherInstallRoot(process.platform === 'win32', hermesHome)?.root, notSource)
  } finally {
    fs.rmSync(base, { recursive: true, force: true })
  }
})
