import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { isolatedElectronArgs, isolateUpdateWindowEnvironment, smokeEnvironment } from '../../tests/install/e2e-assets/smoke-env.mjs'

test('the update window pins its isolated route before Electron requests the singleton lock', (): void => {
  expect(isolatedElectronArgs(
    ['--no-sandbox', '--user-data-dir=/stale/route', '--inspect=0'],
    '/isolated/route',
  )).toEqual(['--user-data-dir=/isolated/route', '--no-sandbox', '--inspect=0'])
})

test.runIf(process.platform !== 'win32')('the launch environment keeps Chromium singleton sockets below the Unix path limit', (): void => {
  const inheritedTemp = path.join(path.sep, 'deep'.repeat(40))

  const env = smokeEnvironment(
    { TEMP: inheritedTemp, TMP: inheritedTemp, TMPDIR: inheritedTemp },
    '/isolated/hermes-home',
    '/isolated/user-data',
  )

  expect(env.TEMP).toBe(env.TMPDIR)
  expect(env.TMP).toBe(env.TMPDIR)
  expect(Buffer.byteLength(path.join(env.TMPDIR, 'scoped_dirXXXXXX', 'SingletonSocket'))).toBeLessThan(108)
})

test('the update window carries Hermes connection state without cloning Chromium state', (): void => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'update-window-userdata-'))
  const userData = path.join(root, 'electron-user-data')
  fs.mkdirSync(userData)
  fs.writeFileSync(path.join(userData, 'connections.json'), '{"primary":"local"}\n')
  fs.writeFileSync(path.join(userData, 'Local State'), 'chromium-owned\n')

  for (const name of ['SingletonLock', 'SingletonSocket', 'SingletonCookie']) {
    fs.symlinkSync(path.join(root, `${name}-target`), path.join(userData, name))
  }

  const captured = {
    HERMES_HOME: path.join(root, 'hermes-home'),
    HERMES_DESKTOP_USER_DATA_DIR: userData,
    PATH: process.env.PATH || '',
  }

  try {
    const isolated = isolateUpdateWindowEnvironment(captured)
    const isolatedUserData = isolated.HERMES_DESKTOP_USER_DATA_DIR

    expect(isolatedUserData).not.toBe(userData)
    expect(captured.HERMES_DESKTOP_USER_DATA_DIR).toBe(userData)
    expect(isolated.HERMES_HOME).toBe(captured.HERMES_HOME)
    expect(fs.readFileSync(path.join(isolatedUserData, 'connections.json'), 'utf8')).toBe('{"primary":"local"}\n')
    expect(fs.existsSync(path.join(isolatedUserData, 'Local State'))).toBe(false)

    for (const name of ['SingletonLock', 'SingletonSocket', 'SingletonCookie']) {
      expect(() => fs.lstatSync(path.join(isolatedUserData, name))).toThrow()
      expect(fs.lstatSync(path.join(userData, name)).isSymbolicLink()).toBe(true)
    }
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
