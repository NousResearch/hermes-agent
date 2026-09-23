import assert from 'node:assert/strict'

import { test } from 'vitest'

import { registerDesktopConnectionApiIpc } from './desktop-connection-api-ipc'

test('a passive registry REST request stays on its named backend', async () => {
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  const dials: string[] = []
  let primaryDials = 0

  const runtime = {
    ipcMain: { handle: (name: string, handler: (...args: any[]) => Promise<any>) => handlers.set(name, handler) },
    ensureRegistryBackend: async (id: string) => {
      dials.push(id)

      return { mode: 'remote', baseUrl: 'https://remote.example', authMode: 'token', token: 'test' }
    },
    ensureBackend: async () => {
      primaryDials += 1
      throw new Error('Primary backend must not be used for a registry request.')
    },
    fetchJsonForBackend: async () => ({ value: 'registry' }),
    desktopProfilePreferences: { afterProfileRequest: () => {} },
    spawnPriorityFrom: () => 'foreground'
  }

  registerDesktopConnectionApiIpc(runtime as any)

  const response = await handlers.get('hermes:api')!(null, {
    connectionId: 'remote-a',
    method: 'GET',
    passive: true,
    path: '/api/status'
  })

  assert.deepEqual(dials, ['remote-a'])
  assert.equal(primaryDials, 0)
  assert.deepEqual(response, { value: 'registry' })
})
