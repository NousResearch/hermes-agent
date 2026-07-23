import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import {
  ensureLinuxUnpackedDeepLink,
  isLinuxUnpackedExecutable,
  registerDeepLinkProtocol
} from './deep-link-registration'

function withTempDir(run) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-deep-link-'))

  try {
    return run(directory)
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
}

function xdgRunner(calls) {
  let current = ''

  return {
    get current() {
      return current
    },
    run(command, args) {
      calls.push([command, args])

      if (command === 'xdg-mime' && args[0] === 'query') {
        return current
      }

      if (command === 'xdg-mime' && args[0] === 'default') {
        current = args[1]
      }

      return ''
    }
  }
}

test('ensureLinuxUnpackedDeepLink installs and verifies the local desktop handler', () =>
  withTempDir(directory => {
    const calls = []
    const runner = xdgRunner(calls)
    const executable = '/opt/Hermes Agent/apps/desktop/release/linux-unpacked/Hermes'

    const result = ensureLinuxUnpackedDeepLink({
      execPath: executable,
      env: { XDG_DATA_HOME: directory },
      homeDir: directory,
      run: runner.run
    })

    assert.equal(result.ok, true)
    assert.equal(result.changed, true)
    assert.equal(runner.current, 'hermes.desktop')
    assert.deepEqual(calls, [
      ['update-desktop-database', [path.join(directory, 'applications')]],
      ['xdg-mime', ['query', 'default', 'x-scheme-handler/hermes']],
      ['xdg-mime', ['default', 'hermes.desktop', 'x-scheme-handler/hermes']],
      ['xdg-mime', ['query', 'default', 'x-scheme-handler/hermes']]
    ])

    const entry = fs.readFileSync(path.join(directory, 'applications', 'hermes.desktop'), 'utf8')
    assert.match(entry, /^Type=Application$/m)
    assert.match(entry, /^Name=Hermes$/m)
    assert.match(entry, /^Exec="\/opt\/Hermes Agent\/apps\/desktop\/release\/linux-unpacked\/Hermes" %U$/m)
    assert.match(entry, /^Terminal=false$/m)
    assert.match(entry, /^MimeType=x-scheme-handler\/hermes;$/m)
  }))

test('ensureLinuxUnpackedDeepLink is idempotent after registration succeeds', () =>
  withTempDir(directory => {
    const calls = []
    const runner = xdgRunner(calls)

    const options = {
      execPath: '/home/user/hermes/apps/desktop/release/linux-unpacked/Hermes',
      env: { XDG_DATA_HOME: directory },
      homeDir: directory,
      run: runner.run
    }

    assert.equal(ensureLinuxUnpackedDeepLink(options).ok, true)
    calls.length = 0

    const second = ensureLinuxUnpackedDeepLink(options)
    assert.equal(second.ok, true)
    assert.equal(second.changed, false)
    assert.deepEqual(calls, [['xdg-mime', ['query', 'default', 'x-scheme-handler/hermes']]])
  }))

test('registerDeepLinkProtocol uses the explicit Linux handler for unpacked builds', () =>
  withTempDir(directory => {
    const runner = xdgRunner([])
    const setAsDefaultProtocolClient = vi.fn(() => false)
    const logs = []

    const registered = registerDeepLinkProtocol({
      protocol: 'hermes',
      platform: 'linux',
      execPath: '/home/user/hermes/apps/desktop/release/linux-unpacked/Hermes',
      defaultApp: false,
      argv: [],
      setAsDefaultProtocolClient,
      log: message => logs.push(message),
      linux: {
        env: { XDG_DATA_HOME: directory },
        homeDir: directory,
        run: runner.run
      }
    })

    assert.equal(registered, true)
    assert.equal(setAsDefaultProtocolClient.mock.calls.length, 0)
    assert.deepEqual(logs, [])
  }))

test('registerDeepLinkProtocol logs a false Electron registration result', () => {
  const logs = []

  const registered = registerDeepLinkProtocol({
    protocol: 'hermes',
    platform: 'linux',
    execPath: '/opt/Hermes/Hermes',
    defaultApp: false,
    argv: [],
    setAsDefaultProtocolClient: () => false,
    log: message => logs.push(message)
  })

  assert.equal(registered, false)
  assert.deepEqual(logs, ['[deeplink] protocol registration failed: setAsDefaultProtocolClient returned false'])
})

test('registerDeepLinkProtocol logs an actionable local Linux registration failure', () =>
  withTempDir(directory => {
    const logs = []

    const registered = registerDeepLinkProtocol({
      protocol: 'hermes',
      platform: 'linux',
      execPath: '/home/user/hermes/apps/desktop/release/linux-unpacked/Hermes',
      defaultApp: false,
      argv: [],
      setAsDefaultProtocolClient: vi.fn(() => false),
      log: message => logs.push(message),
      linux: {
        env: { XDG_DATA_HOME: directory },
        homeDir: directory,
        run: command => {
          throw new Error(`${command} is unavailable`)
        }
      }
    })

    assert.equal(registered, false)
    assert.equal(logs.length, 1)
    assert.match(logs[0], /could not register x-scheme-handler\/hermes with hermes\.desktop/)
    assert.match(logs[0], /xdg-mime is unavailable/)
  }))

test('isLinuxUnpackedExecutable only matches the local electron-builder layout', () => {
  assert.equal(isLinuxUnpackedExecutable('/repo/apps/desktop/release/linux-unpacked/Hermes'), true)
  assert.equal(isLinuxUnpackedExecutable('/opt/Hermes/Hermes'), false)
  assert.equal(isLinuxUnpackedExecutable('/repo/apps/desktop/release/linux-unpacked-old/Hermes'), false)
})
