import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import { createDesktopInstallHomeRuntime } from './desktop-install-home-runtime'

test('install stamp and isolated desktop home resolve from the selected app roots', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-install-home-'))

  try {
    vi.stubEnv('HERMES_HOME', '')

    const build = path.join(root, 'build')

    fs.mkdirSync(build)
    fs.writeFileSync(
      path.join(build, 'install-stamp.json'),
      JSON.stringify({ schemaVersion: 1, commit: '1234567890abcdef', branch: 'main', source: 'test' })
    )

    const runtime = createDesktopInstallHomeRuntime({
      APP_ROOT: root,
      USER_DATA_OVERRIDE: path.join(root, 'user-data'),
      IS_WINDOWS: false,
      app: { getPath: () => root },
      directoryExists: candidate => fs.existsSync(candidate) && fs.statSync(candidate).isDirectory(),
      normalizeHermesHomeRoot: value => path.resolve(value),
      readWindowsUserEnvVar: () => null
    })

    expect(runtime.loadInstallStamp()).toMatchObject({ commit: '1234567890abcdef', source: 'test' })
    expect(runtime.resolveHermesHome()).toBe(path.join(root, 'user-data', 'hermes-home'))
  } finally {
    vi.unstubAllEnvs()
    expect(path.resolve(root).startsWith(path.resolve(os.tmpdir()) + path.sep)).toBe(true)

    fs.rmSync(root, { recursive: true, force: true })
  }
})
