import assert from 'node:assert/strict'

import { test } from 'vitest'

import { type ConnectionRegistry, reconcileAppliedGlobalConnection } from './connection-registry'
import { resolveDesktopConnectionRequest } from './desktop-profile'
import {
  appliedPrimaryWindowRoute,
  applyPrimaryConnectionRoute,
  normalizeWindowConnectionRoute,
  registrySshPoolScopeByConnectionId,
  registrySshScopeForWindowRoute,
  WindowConnectionRouteRegistry
} from './window-connection-route'

test('normalizes an exact registry-scoped connection and profile', () => {
  assert.deepEqual(
    normalizeWindowConnectionRoute({
      connectionId: 'source-b',
      profile: 'research',
      registryScoped: true
    }),
    {
      connectionId: 'source-b',
      profile: 'research',
      registryScoped: true
    }
  )
})

test('keeps legacy/profile-only routes distinct from registry identities', () => {
  assert.deepEqual(normalizeWindowConnectionRoute({ profile: 'work' }), {
    connectionId: null,
    profile: 'work',
    registryScoped: false
  })
})

test('preserves a registry connection when no profile is selected', () => {
  assert.deepEqual(
    normalizeWindowConnectionRoute({
      connectionId: 'source-b',
      registryScoped: true
    }),
    {
      connectionId: 'source-b',
      profile: undefined,
      registryScoped: true
    }
  )
})

test('isolates active routes by webContents id', () => {
  const routes = new WindowConnectionRouteRegistry()

  routes.set(11, { connectionId: 'source-a', profile: 'default', registryScoped: true })
  routes.set(22, { connectionId: 'source-b', profile: 'worker', registryScoped: true })

  assert.equal(routes.get(11)?.connectionId, 'source-a')
  assert.equal(routes.get(22)?.connectionId, 'source-b')

  routes.delete(11)

  assert.equal(routes.get(11), null)
  assert.equal(routes.get(22)?.connectionId, 'source-b')
})

test('invalid publications clear only the sender route', () => {
  const routes = new WindowConnectionRouteRegistry()

  routes.set(11, { connectionId: 'source-a', profile: 'default', registryScoped: true })
  routes.set(22, { connectionId: 'source-b', profile: 'worker', registryScoped: true })

  routes.set(11, null)

  assert.equal(routes.get(11), null)
  assert.equal(routes.get(22)?.connectionId, 'source-b')
})

test('routes a non-primary SSH connection independently from another window', () => {
  const registry = {
    primary: 'source-a',
    connections: [
      { id: 'source-a', kind: 'ssh' },
      { id: 'source-b', kind: 'ssh' },
      { id: 'source-c', kind: 'remote' }
    ]
  } as never

  const routes = new WindowConnectionRouteRegistry()

  routes.set(11, {
    connectionId: 'source-b',
    profile: 'worker',
    registryScoped: true
  })
  routes.set(22, {
    connectionId: 'source-c',
    profile: 'default',
    registryScoped: true
  })

  assert.equal(registrySshScopeForWindowRoute(routes.get(11), registry), 'conn:source-b::worker')
  assert.equal(registrySshScopeForWindowRoute(routes.get(22), registry), null)
})

test('uses the canonical default profile scope when a registry SSH route has no profile', () => {
  const registry = {
    primary: 'source-a',
    connections: [
      { id: 'source-a', kind: 'ssh' },
      { id: 'source-b', kind: 'ssh' }
    ]
  } as never

  assert.equal(
    registrySshScopeForWindowRoute(
      {
        connectionId: 'source-b',
        profile: undefined,
        registryScoped: true
      },
      registry
    ),
    'conn:source-b::default'
  )
})

test('recovers the bootstrap pool key a registry SSH tunnel was published under', () => {
  const pool = new Map<string, any>([['', { registryConnectionId: 'source-b', ssh: { alive: true } }]])

  assert.equal(registrySshPoolScopeByConnectionId(pool, 'source-b'), '')
})

test('does not match another connection, an unlabelled entry, or a torn-down tunnel', () => {
  const pool = new Map<string, any>([
    ['research', { registryConnectionId: 'source-a', ssh: { alive: true } }],
    ['', { registryConnectionId: '', ssh: { alive: true } }],
    ['worker', { registryConnectionId: 'source-b' }]
  ])

  assert.equal(registrySshPoolScopeByConnectionId(pool, 'source-b'), null)
  assert.equal(registrySshPoolScopeByConnectionId(pool, 'source-c'), null)
})

test('a primary apply re-points the window route so its re-dial leaves the gateway it just left', () => {
  // #92352: the window was on a registered remote and the config apply chose
  // This device. A profile-less re-dial must no longer inherit that old source.
  const routes = new WindowConnectionRouteRegistry()
  const windowId = 11

  routes.set(windowId, { connectionId: 'macmini', profile: 'default', registryScoped: true })

  const previousRegistry: ConnectionRegistry = {
    version: 2,
    primary: 'macmini',
    launchMode: 'primary',
    lastUsed: 'macmini',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      { id: 'macmini', kind: 'remote', label: 'Mac Mini', url: 'https://macmini.example', authMode: 'token' }
    ]
  }

  const appliedRegistry = reconcileAppliedGlobalConnection(previousRegistry, { mode: 'local' })

  assert.equal(appliedRegistry.primary, 'local')
  assert.deepEqual(resolveDesktopConnectionRequest(undefined, routes.get(windowId), 'default'), {
    connectionId: 'macmini',
    profile: 'default'
  })

  const order: string[] = []
  let redial: ReturnType<typeof resolveDesktopConnectionRequest> | null = null

  applyPrimaryConnectionRoute({
    routeRegistry: routes,
    webContentsId: windowId,
    registry: appliedRegistry,
    fallbackProfile: 'default',
    recordRoute: route => {
      order.push('record')
      routes.set(windowId, route)
    },
    notifyApplied: () => {
      order.push('notify')
      redial = resolveDesktopConnectionRequest(undefined, routes.get(windowId), 'default')
    }
  })

  assert.deepEqual(redial, { connectionId: null, profile: 'default' })
  assert.deepEqual(order, ['record', 'notify'])
})

test('a remote apply stays registry-scoped to the applied source and keeps the viewed profile', () => {
  const registry = {
    primary: 'macmini',
    connections: [
      { id: 'local', kind: 'local' },
      { id: 'macmini', kind: 'remote' }
    ]
  } as never

  const applied = appliedPrimaryWindowRoute(registry, 'work')

  assert.deepEqual(applied, { connectionId: 'macmini', profile: 'work', registryScoped: true })
  // A window launched from this route afterwards inherits the applied source.
  assert.deepEqual(resolveDesktopConnectionRequest(undefined, applied, 'default'), {
    connectionId: 'macmini',
    profile: 'work'
  })
})

test('falls back to the canonical profile when the applied window had no route to keep', () => {
  assert.deepEqual(appliedPrimaryWindowRoute({ primary: 'local' } as never, undefined), {
    connectionId: null,
    profile: 'default',
    registryScoped: false
  })
  assert.deepEqual(appliedPrimaryWindowRoute({ primary: '' } as never, '  '), {
    connectionId: null,
    profile: 'default',
    registryScoped: false
  })
})
