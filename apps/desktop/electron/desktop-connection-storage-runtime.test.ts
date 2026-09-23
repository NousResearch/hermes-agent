import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { createDesktopConnectionStorageRuntime } from './desktop-connection-storage-runtime'

test('a corrupt registry is retained while the connection store returns a local route', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-connection-store-'))
  const registryPath = path.join(dir, 'connections.json')
  const original = '{"connections": [broken]}'

  try {
    fs.writeFileSync(registryPath, original)

    const runtime = createDesktopConnectionStorageRuntime({
      app: { getPath: () => dir },
      safeStorage: { isEncryptionAvailable: () => false },
      session: { defaultSession: {} },
      connectionConfigPath: path.join(dir, 'connection.json'),
      connectionsRegistryPath: registryPath,
      profileNameRe: /^[A-Za-z0-9_-]+$/,
      nativeTokenStoreIo: () => ({
        encrypt: () => ({ encoding: 'plain', value: '' }),
        decrypt: () => '',
        readStoreText: () => '{}',
        writeStoreText: () => {},
        rememberLog: () => {}
      }),
      rememberLog: () => {},
      assertCanMutateRegistryConnection: () => {},
      stopRegistryConnectionBackends: async () => {},
      broadcastConnectionsChanged: () => {}
    })

    const registry = runtime.readDesktopConnectionsRegistry()

    assert.equal(registry.connections.some(connection => connection.kind === 'local'), true)
    assert.equal(fs.readFileSync(registryPath, 'utf8'), original)
    const sidecars = fs.readdirSync(dir).filter(name => name.startsWith('connections.json.corrupt-'))
    assert.equal(sidecars.length, 1)
    assert.equal(fs.readFileSync(path.join(dir, sidecars[0]), 'utf8'), original)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
