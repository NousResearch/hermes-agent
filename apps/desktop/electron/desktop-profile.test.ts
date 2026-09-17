import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  createDesktopProfilePreferences,
  resolveDesktopConnectionRequest,
  resolveDesktopWindowRoute
} from './desktop-profile'
import { WindowConnectionRouteRegistry } from './window-connection-route'

test('failed authoritative writes leave the previous default and listeners untouched', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-profile-write-'))
  const target = path.join(root, 'active-profile.json')
  const changes: unknown[] = []

  const preferences = createDesktopProfilePreferences(target, {
    onDefaultChanged: route => changes.push(route),
    validateRoute: route => {
      if (route.connectionId === 'missing') {
        throw new Error('Connection was removed')
      }
    }
  })

  try {
    const original = { connectionId: null, profile: 'work' }
    preferences.setDefault(original)

    for (const invalid of [
      null,
      {},
      { connectionId: 'missing', profile: 'work' },
      { connectionId: null, profile: ' work ' }
    ]) {
      assert.throws(() => preferences.setDefault(invalid))
      assert.deepEqual(preferences.getDefault(), original)
    }

    fs.mkdirSync(`${target}.tmp`)
    assert.throws(() => preferences.setDefault({ connectionId: 'remote', profile: 'personal' }))
    assert.deepEqual(preferences.getDefault(), original)
    assert.deepEqual(changes, [original])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('explicit routes are strict and never fall back to a different source window', () => {
  const routes = new WindowConnectionRouteRegistry()
  routes.set(1, { connectionId: 'remote-a', profile: 'work', registryScoped: true })
  routes.set(2, { connectionId: 'remote-b', profile: 'work', registryScoped: true })
  const fallback = { connectionId: null, profile: 'default' }
  const explicit = { connectionId: 'remote-c', profile: 'personal' }

  assert.deepEqual(resolveDesktopWindowRoute(undefined, routes.get(1), fallback), {
    connectionId: 'remote-a',
    profile: 'work'
  })
  assert.deepEqual(resolveDesktopWindowRoute(explicit, routes.get(1), fallback), explicit)
  assert.deepEqual(resolveDesktopWindowRoute(undefined, routes.get(2), fallback), {
    connectionId: 'remote-b',
    profile: 'work'
  })
  assert.deepEqual(resolveDesktopWindowRoute(undefined, routes.get(1), fallback), {
    connectionId: 'remote-a',
    profile: 'work'
  })
  assert.deepEqual(routes.get(1), { connectionId: 'remote-a', profile: 'work', registryScoped: true })
  assert.throws(() => resolveDesktopWindowRoute({ profile: 'work' }, routes.get(1), fallback))
  assert.throws(() => resolveDesktopWindowRoute({ connectionId: null, profile: '../work' }, routes.get(1), fallback))
  assert.throws(() => resolveDesktopWindowRoute({ connectionId: '', profile: 'work' }, routes.get(1), fallback))
})

test('boot and reconnect retain the window route rather than a later global default or another window', () => {
  const routeA = { connectionId: 'remote-a', profile: 'work', registryScoped: true }
  const routeB = { connectionId: null, profile: 'personal', registryScoped: false }

  for (const route of [routeA, routeB, routeA]) {
    assert.deepEqual(resolveDesktopConnectionRequest(undefined, route, 'last-used'), {
      connectionId: route.connectionId,
      profile: route.profile
    })
    assert.deepEqual(resolveDesktopConnectionRequest(route.profile, route, 'last-used'), {
      connectionId: null,
      profile: route.profile
    })
  }

  assert.deepEqual(resolveDesktopConnectionRequest('other', routeA, 'last-used'), {
    connectionId: null,
    profile: 'other'
  })
  assert.deepEqual(resolveDesktopConnectionRequest(undefined, null, 'last-used'), {
    connectionId: null,
    profile: 'last-used'
  })
})

test('an explicit default survives last-used profile writes and app restarts, isolated by desktop home', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-profile-'))

  try {
    const homeA = path.join(root, 'a', 'active-profile.json')
    const homeB = path.join(root, 'b', 'active-profile.json')
    const changes: unknown[] = []
    const a = createDesktopProfilePreferences(homeA, { onDefaultChanged: route => changes.push(route) })
    const b = createDesktopProfilePreferences(homeB)
    const route = { connectionId: 'remote-work', profile: 'work' }

    assert.equal(a.getDefault(), null)
    assert.equal(a.remember('personal'), 'personal')
    assert.deepEqual(a.setDefault(route), route)
    a.remember('other')
    b.setDefault({ connectionId: null, profile: 'personal' })

    assert.equal(a.readActive(), 'other')
    assert.deepEqual(createDesktopProfilePreferences(homeA).getDefault(), route)
    assert.deepEqual(b.getDefault(), { connectionId: null, profile: 'personal' })
    assert.deepEqual(a.getDefault(), route)
    assert.deepEqual(changes, [route])
    a.afterProfileRequest(
      'remote-work',
      { method: 'PATCH', path: '/api/profiles/work', body: { new_name: 'ignored' } },
      { ok: false },
      'remote'
    )
    assert.deepEqual(a.getDefault(), route)
    a.afterProfileRequest(
      'remote-work',
      { method: 'PATCH', path: '/api/profiles/work', body: { new_name: 'renamed' } },
      { ok: true },
      'remote'
    )
    assert.deepEqual(a.getDefault(), { ...route, profile: 'renamed' })
    a.afterProfileRequest('remote-work', { method: 'DELETE', path: '/api/profiles/renamed' }, { ok: true }, 'remote')
    assert.equal(a.getDefault(), null)
    a.setDefault(route)
    a.profileChanged('another-source', 'work', 'renamed', 'remote')
    assert.deepEqual(a.getDefault(), route)
    a.profileChanged(route.connectionId, route.profile, 'renamed', 'remote')
    assert.deepEqual(a.getDefault(), { ...route, profile: 'renamed' })
    a.profileChanged(route.connectionId, 'renamed', null, 'remote')
    assert.equal(a.getDefault(), null)
    a.setDefault(route)
    a.connectionRemoved(route.connectionId)
    assert.equal(createDesktopProfilePreferences(homeA).getDefault(), null)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test.each([null, 'local'])(
  'successful local profile changes through %s retarget the saved startup profile',
  connectionId => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-profile-change-'))
    const target = path.join(root, 'active-profile.json')
    const preferences = createDesktopProfilePreferences(target)

    try {
      const defaultRoute = { connectionId: 'remote-work', profile: 'local-old' }
      preferences.setDefault(defaultRoute)
      preferences.remember('local-old')
      preferences.afterProfileRequest(
        connectionId,
        { method: 'DELETE', path: '/api/profiles/local-old' },
        { ok: false },
        'local'
      )
      assert.equal(preferences.readActive(), 'local-old')

      preferences.afterProfileRequest(
        'remote-work',
        { method: 'DELETE', path: '/api/profiles/local-old' },
        { ok: true },
        'remote'
      )
      assert.equal(preferences.readActive(), 'local-old')
      preferences.setDefault(defaultRoute)

      preferences.afterProfileRequest(
        connectionId,
        { method: 'PATCH', path: '/api/profiles/local-old', body: { new_name: 'local-new' } },
        { ok: true },
        'local'
      )
      const restarted = createDesktopProfilePreferences(target)
      assert.equal(restarted.readActive(), 'local-new')
      assert.deepEqual(restarted.getDefault(), defaultRoute)

      restarted.afterProfileRequest(
        connectionId,
        { method: 'DELETE', path: '/api/profiles/local-new' },
        { ok: true },
        'local'
      )
      assert.equal(createDesktopProfilePreferences(target).readActive(), 'default')
      assert.deepEqual(restarted.getDefault(), defaultRoute)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  }
)
