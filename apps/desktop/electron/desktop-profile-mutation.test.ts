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
  })

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

// ---------------------------------------------------------------------------
// Regression: null vs 'local' connectionId mismatch on rename/delete
// ---------------------------------------------------------------------------
//
// The renderer stores defaultRoute with connectionId:null (legacy local path).
// When a rename or delete goes through the registry IPC path, afterProfileRequest
// receives connectionId:'local'. The old guard used strict equality, so
// null !== 'local' → profileChanged bailed out without updating defaultRoute.
// On the next restart, connectDesktopProfileRoute(defaultRoute) tried the old
// profile name and the backend crashed with "Profile does not exist".

for (const method of ['PATCH', 'DELETE']) {
  test(`${method}: null-stored defaultRoute is updated when afterProfileRequest fires with connectionId 'local'`, () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-profile-null-local-'))
    const target = path.join(root, 'active-profile.json')
    const preferences = createDesktopProfilePreferences(target)

    const request = {
      connectionId: 'local',
      method,
      path: '/api/profiles/work',
      body: method === 'PATCH' ? { new_name: 'renamed' } : undefined
    }

    try {
      // Simulate what the renderer does: stores defaultRoute with connectionId:null
      // (the legacy local path — namedProfileConnectionId is null on standard setups).
      preferences.remember('work')
      preferences.setDefault({ connectionId: null, profile: 'work' })

      // Rename/delete comes through the registry IPC path with connectionId:'local'.
      preferences.afterProfileRequest('local', request, { ok: true }, 'local')

      const restarted = createDesktopProfilePreferences(target)

      if (method === 'PATCH') {
        // defaultRoute must be updated to the new name so boot uses 'renamed', not 'work'.
        assert.deepEqual(restarted.getDefault(), { connectionId: null, profile: 'renamed' })
        assert.equal(restarted.readActive(), 'renamed')
      } else {
        // defaultRoute must be cleared so boot falls back to startHermes().
        assert.equal(restarted.getDefault(), null)
        assert.equal(restarted.readActive(), 'default')
      }
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
}
