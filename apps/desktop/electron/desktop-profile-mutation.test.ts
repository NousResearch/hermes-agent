import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { apiRequestRegistryConnectionId } from './connection-config'
import { createDesktopProfilePreferences } from './desktop-profile'

const sources = [
  { label: 'legacy local', connectionId: null, mode: 'local' },
  { label: 'registry local', connectionId: 'local', mode: 'local' },
  { label: 'legacy remote', connectionId: null, mode: 'remote' },
  { label: 'legacy SSH', connectionId: null, mode: 'ssh' },
  { label: 'registry remote', connectionId: 'remote-work', mode: 'remote' },
  { label: 'registry SSH', connectionId: 'ssh-work', mode: 'ssh' },
  { label: 'registry cloud', connectionId: 'cloud-work', mode: 'cloud' }
]

for (const method of ['PATCH', 'DELETE']) {
  // These cases create, sync-write, reopen and recursively remove two files on
  // the native filesystem. Windows antivirus and a saturated CI worker can
  // delay one synchronous operation past Vitest's 5s default (5.49s observed
  // in the full Electron suite); the ownership assertions remain unchanged.
  test.each(sources)(`${method} on $label preserves startup ownership across restart`, source => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-profile-owner-'))
    const target = path.join(root, 'active-profile.json')
    const otherTarget = path.join(root, 'other', 'active-profile.json')
    const preferences = createDesktopProfilePreferences(target)
    const other = createDesktopProfilePreferences(otherTarget)

    const request = {
      connectionId: source.connectionId,
      method,
      path: '/api/profiles/work',
      body: method === 'PATCH' ? { new_name: 'renamed' } : undefined
    }

    const connectionId = apiRequestRegistryConnectionId(request)
    const route = { connectionId, profile: 'work' }

    try {
      preferences.remember('work')
      preferences.setDefault(route)
      other.remember('work')
      other.setDefault(route)
      preferences.afterProfileRequest(connectionId, request, { ok: true }, source.mode)

      const restarted = createDesktopProfilePreferences(target)
      assert.equal(
        restarted.readActive(),
        source.mode === 'local' ? (method === 'PATCH' ? 'renamed' : 'default') : 'work'
      )
      assert.deepEqual(restarted.getDefault(), method === 'PATCH' ? { ...route, profile: 'renamed' } : null)
      assert.equal(createDesktopProfilePreferences(otherTarget).readActive(), 'work')
      assert.deepEqual(other.getDefault(), route)
      assert.equal(preferences.readActive(), restarted.readActive())
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  }, 15_000)

  test.each([null, 'local'])(`${method} failures through %s preserve both preferences`, connectionId => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-profile-failure-'))
    const target = path.join(root, 'active-profile.json')
    const preferences = createDesktopProfilePreferences(target)
    const route = { connectionId, profile: 'work' }
    const request = { method, path: '/api/profiles/work', body: { new_name: 'renamed' } }

    try {
      preferences.remember('work')
      preferences.setDefault(route)
      const original = fs.readFileSync(target, 'utf8')

      for (const failure of [{ ok: false }, { success: false }, { error: 'Permission denied' }]) {
        preferences.afterProfileRequest(connectionId, request, failure, 'local')
        assert.equal(fs.readFileSync(target, 'utf8'), original)
      }

      preferences.afterProfileRequest(connectionId, { ...request, path: '/api/profiles/other' }, { ok: true }, 'local')
      assert.equal(fs.readFileSync(target, 'utf8'), original)

      fs.mkdirSync(`${target}.tmp`)
      assert.throws(() => preferences.afterProfileRequest(connectionId, request, { ok: true }, 'local'))
      assert.equal(fs.readFileSync(target, 'utf8'), original)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
}
