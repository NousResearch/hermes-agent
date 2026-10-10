import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  assertSpawnableProfileName,
  createDesktopProfilePreferences,
  requireDesktopProfileName,
  resolveDesktopConnectionRequest,
  resolveDesktopWindowLaunch,
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
  // Only an explicit route pins the window's New-session default; an inherited
  // or fallback route seeds boot alone.
  assert.deepEqual(resolveDesktopWindowLaunch(explicit, routes.get(1), fallback), { ...explicit, profileWindow: true })
  assert.deepEqual(resolveDesktopWindowLaunch(undefined, routes.get(1), fallback), {
    connectionId: 'remote-a',
    profile: 'work',
    profileWindow: false
  })
  assert.deepEqual(resolveDesktopWindowLaunch(undefined, null, fallback), { ...fallback, profileWindow: false })
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

// GHSA-84j8-xmx8-jghv: a renderer-supplied profile name reaches `--profile <name>`
// on a backend spawn that can run with shell: true (Windows .cmd/.bat shims), so
// the connection path must reject anything that is not a canonical name — not
// only the persisted profile setter.
const HOSTILE_PROFILE_NAMES = [
  'work; calc.exe',
  'work && whoami',
  'work | more',
  'work`whoami`',
  'work$(whoami)',
  'work%PATH%',
  'work > out.txt',
  'work\nserve',
  '--profile',
  '-work',
  'my profile',
  'Work',
  '../default',
  'a'.repeat(65)
]

test('requireDesktopProfileName accepts canonical names and treats blank as unset', () => {
  assert.equal(requireDesktopProfileName('work'), 'work')
  assert.equal(requireDesktopProfileName('  work  '), 'work')
  assert.equal(requireDesktopProfileName('default'), 'default')
  assert.equal(requireDesktopProfileName('a-b_c9'), 'a-b_c9')
  assert.equal(requireDesktopProfileName(''), null)
  assert.equal(requireDesktopProfileName('   '), null)
  assert.equal(requireDesktopProfileName(null), null)
  assert.equal(requireDesktopProfileName(undefined), null)
})

test('requireDesktopProfileName rejects shell metacharacters, flags, whitespace and non-strings', () => {
  for (const name of HOSTILE_PROFILE_NAMES) {
    assert.throws(() => requireDesktopProfileName(name), /Invalid profile name/, name)
  }

  for (const value of [42, {}, [], { profile: 'work' }, true]) {
    assert.throws(() => requireDesktopProfileName(value), /Invalid profile name/)
  }
})

test('the connection request path rejects invalid profile names instead of forwarding them', () => {
  for (const name of HOSTILE_PROFILE_NAMES) {
    assert.throws(() => resolveDesktopConnectionRequest(name, null, 'last-used'), /Invalid profile name/, name)
  }

  // Valid names still resolve exactly as before; blank still falls back.
  assert.deepEqual(resolveDesktopConnectionRequest(' work ', null, 'last-used'), {
    connectionId: null,
    profile: 'work'
  })
  assert.deepEqual(resolveDesktopConnectionRequest('', null, 'last-used'), {
    connectionId: null,
    profile: 'last-used'
  })
})

test('assertSpawnableProfileName only passes an exact canonical name through to child argv', () => {
  assert.equal(assertSpawnableProfileName('work'), 'work')
  assert.equal(assertSpawnableProfileName('default'), 'default')

  for (const value of [...HOSTILE_PROFILE_NAMES, ' work', 'work ', '', null, undefined, 7]) {
    assert.throws(() => assertSpawnableProfileName(value), /Refusing to start a backend/, String(value))
  }
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
