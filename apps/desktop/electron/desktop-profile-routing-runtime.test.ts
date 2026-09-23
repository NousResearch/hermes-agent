import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { createDesktopProfileRoutingRuntime } from './desktop-profile-routing-runtime'

test('profile preferences validate the live registry and broadcast the explicit default', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-profile-routing-'))
  const target = path.join(root, 'active-profile.json')
  const messages: unknown[] = []
  let registeredIds = ['local']

  try {
    const runtime = createDesktopProfileRoutingRuntime({
      configPath: target,
      hermesHome: root,
      profileNameRe: /^[a-z0-9][a-z0-9_-]{0,63}$/,
      readDesktopConnectionsRegistry: () => ({ connections: registeredIds.map(id => ({ id })) }),
      readDesktopConnectionConfig: () => ({ profiles: { work: {} } }),
      primaryProfileKey: () => 'default',
      globalRemoteActive: () => false,
      primaryBackendIsRemote: () => false,
      getIsolatedBackend: () => true,
      BrowserWindow: {
        getAllWindows: () => [{ webContents: { isDestroyed: () => false, send: (...args: unknown[]) => messages.push(args) } }]
      },
      writeFileAtomic: () => undefined
    })

    const route = { connectionId: 'remote', profile: 'work' }
    assert.throws(() => runtime.desktopProfilePreferences.setDefault(route), /No connection/)
    registeredIds = ['local', 'remote']
    assert.deepEqual(runtime.desktopProfilePreferences.setDefault(route), route)
    assert.deepEqual(messages, [['hermes:profile:default:changed', route]])
    assert.equal(runtime.writeActiveDesktopProfile('work'), 'work')
    assert.equal(runtime.readActiveDesktopProfile(), 'work')
    assert.deepEqual(runtime.profileRouteOptions('work', { method: 'GET', path: '/health' }), {
      backendProfile: undefined,
      globalRemote: false,
      primaryProfile: 'default',
      profileRemoteOverride: false,
      primaryRemoteActive: false,
      ownEntry: true,
      isolatedBackend: true,
      requestMethod: 'GET',
      requestPath: '/health'
    })
  } finally {
    assert.equal(path.basename(root).startsWith('desktop-profile-routing-'), true)
    assert.equal(path.dirname(root), os.tmpdir())
    fs.rmSync(root, { recursive: true, force: true })
  }
})
