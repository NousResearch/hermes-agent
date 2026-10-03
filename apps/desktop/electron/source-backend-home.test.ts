import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import { resolveSourceInstallationBackend } from './source-backend'

test.skipIf(process.platform !== 'win32')(
  'a source probe retains a usable Python home after a depleted GUI launch',
  async () => {
    const python = process.env.HERMES_PYTHON || 'python'
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-depleted-home-'))
    const home = path.join(root, 'user home')
    fs.mkdirSync(home)
    fs.mkdirSync(path.join(root, 'hermes_cli'))
    fs.writeFileSync(path.join(root, 'hermes_cli', 'main.py'), '')
    fs.mkdirSync(path.join(root, '.hermes', 'bin'), { recursive: true })
    fs.writeFileSync(
      path.join(root, '.hermes', 'bin', 'hermes.cmd'),
      `@echo off\r\n"${python}" -I -c "from pathlib import Path; import os; assert Path.home() == Path(os.environ['EXPECTED_HOME'])"\r\n`
    )
    // Node resolves the account home through the OS when USERPROFILE is absent.
    // Substitute a fixture home so no probe can touch the real user's state.
    const homedir = vi.spyOn(os, 'homedir').mockReturnValue(home)
    try {
      for (const key of ['USERPROFILE', 'HOMEDRIVE', 'HOMEPATH']) vi.stubEnv(key, undefined)
      const env = {
        ...process.env,
        USERPROFILE: undefined,
        HOMEDRIVE: undefined,
        HOMEPATH: undefined,
        HOME: home,
        APPDATA: path.join(home, 'Roaming'),
        LOCALAPPDATA: path.join(home, 'Local'),
        HERMES_HOME: path.join(home, 'hermes'),
        EXPECTED_HOME: home
      }
      const control = spawnSync(python, ['-I', '-c', 'from pathlib import Path; Path.home()'], {
        env,
        encoding: 'utf8',
        windowsHide: true,
        timeout: 15_000
      })
      assert.equal(control.status, 1)
      assert.equal(control.error, undefined)
      assert.match(control.stderr, /Could not determine home directory/)
      const backend = await resolveSourceInstallationBackend(root, ['serve'], {
        hermesHome: env.HERMES_HOME,
        env
      })
      assert.ok(backend, 'healthy launcher must not be rejected because Python cannot determine its home')
      assert.equal(backend.env.USERPROFILE, home)
    } finally {
      homedir.mockRestore()
      vi.unstubAllEnvs()
      fs.rmSync(root, { recursive: true, force: true })
    }
  },
  30_000
)
