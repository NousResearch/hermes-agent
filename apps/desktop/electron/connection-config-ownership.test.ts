/** Focused owner-routing contracts for electron/connection-config.ts. */

import assert from 'node:assert/strict'
import http from 'node:http'
import type { AddressInfo } from 'node:net'

import { test, vi } from 'vitest'

import {
  apiRequestRegistryConnectionId,
  bindSessionRowsToRoute,
  createSessionRouteBinding,
  dispatchLegacySessionRequest,
  resolveLegacyApiConnection,
  resolveProfileApiRequest,
  resolveRegistryApiConnection
} from './connection-config'

const mutationContract = [
  { action: 'archive', body: { archived: true }, method: 'PATCH', targetId: 'session-live' },
  { action: 'unarchive', body: { archived: false }, method: 'PATCH', targetId: 'session-live' },
  { action: 'delete', body: null, method: 'DELETE', targetId: 'session-live' },
  { action: 'rename', body: { title: 'Renamed' }, method: 'PATCH', targetId: 'session-live' },
  { action: 'pin', body: { pinned: true }, method: 'PATCH', targetId: 'session-root' },
  { action: 'unread', body: { unread: true }, method: 'PATCH', targetId: 'session-live' }
] as const satisfies readonly {
  action: string
  body: null | Readonly<Record<string, boolean | string>>
  method: 'DELETE' | 'PATCH'
  targetId: string
}[]

test('legacy session rows inherit the trusted route profile before they are bound', () => {
  const connection = { mode: 'remote', baseUrl: 'https://legacy.example', token: 'legacy-token' }
  const [row] = bindSessionRowsToRoute([{ id: 'legacy-row' }], connection, 'research')

  assert.equal(row.profile, 'research')
  assert.equal(row._desktop_route_binding, createSessionRouteBinding('legacy-row', 'research', connection))
})

test('legacy Settings resolution retains its HTTP origin across awaits and rejects route replacement', async () => {
  const received: string[] = []

  const server = http.createServer((request, response) => {
    received.push(`${request.method} ${request.url}`)
    response.setHeader('Content-Type', 'application/json')
    response.end(JSON.stringify({ ok: true }))
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  const legacyConnection = {
    mode: 'remote',
    baseUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
    token: 'fixture'
  }

  const request = { connectionId: undefined, profile: 'worker', path: '/api/config', legacyConnection }
  let resolve!: (connection: typeof legacyConnection) => void
  let backend = Promise.resolve(legacyConnection)
  const ensureBackend = vi.fn((_profile: string | null) => backend)

  const dispatch = async (method: string) => {
    assert.equal(apiRequestRegistryConnectionId(request), null)

    const route = resolveProfileApiRequest(request.profile, request.path, {
      globalRemote: true,
      primaryProfile: 'default'
    })

    const connection = await resolveLegacyApiConnection(request, route.backendProfile, ensureBackend)

    return (await fetch(`${connection.baseUrl}${route.requestPath}`, { method })).json()
  }

  try {
    await dispatch('GET')
    backend = new Promise(yes => {
      resolve = yes
    })
    const write = dispatch('PUT')
    resolve(legacyConnection)
    await write
    assert.deepEqual(received, ['GET /api/config?profile=worker', 'PUT /api/config?profile=worker'])
    assert.ok(ensureBackend.mock.calls.every(args => args[0] === null))

    // A connection apply while resolution awaits may replace the v1 route. No local write.
    for (const replacement of [
      { ...legacyConnection, mode: 'local' },
      { ...legacyConnection, baseUrl: 'https://replacement.invalid' },
      { ...legacyConnection, token: 'replacement' },
      { ...legacyConnection, headers: { 'Cf-Access-Client-Id': 'replacement' } }
    ]) {
      backend = new Promise(yes => {
        resolve = yes
      })
      const stale = dispatch('PUT')
      resolve(replacement)
      await assert.rejects(stale, /Backend changed/)
    }

    assert.equal(received.length, 2)
    await assert.rejects(
      resolveLegacyApiConnection({ legacyConnection: null }, null, () => Promise.resolve(legacyConnection)),
      /Backend changed/
    )
  } finally {
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('legacy Settings resolves a selected profile override without comparing it to the foreground owner', async () => {
  const foreground = { mode: 'remote', baseUrl: 'https://foreground.example', token: 'foreground-token' }
  const override = { mode: 'remote', baseUrl: 'https://override.example', token: 'override-token' }
  const ensureBackend = vi.fn(async profile => (profile === 'default' ? foreground : override))

  const request = {
    legacyConnection: foreground,
    legacyConnectionProfile: 'default',
    profile: 'worker',
    path: '/api/config'
  }

  assert.equal(await resolveLegacyApiConnection(request, 'worker', ensureBackend), override)
  assert.deepEqual(ensureBackend.mock.calls, [
    ['default', { passive: undefined }],
    ['worker', { passive: undefined }],
    ['default', { passive: undefined }]
  ])
})

test('legacy Settings rejects a selected profile override when its foreground owner changes while resolving', async () => {
  const foreground = { mode: 'remote', baseUrl: 'https://foreground.example', token: 'foreground-token' }
  const replacement = { ...foreground, token: 'replacement-token' }
  const override = { mode: 'remote', baseUrl: 'https://override.example', token: 'override-token' }
  let foregroundCalls = 0

  const ensureBackend = vi.fn(async profile => {
    if (profile === 'worker') {
      return override
    }

    foregroundCalls += 1

    return foregroundCalls === 1 ? foreground : replacement
  })

  await assert.rejects(
    resolveLegacyApiConnection(
      {
        legacyConnection: foreground,
        legacyConnectionProfile: 'default',
        profile: 'worker',
        path: '/api/config'
      },
      'worker',
      ensureBackend
    ),
    /Backend changed/
  )
})

test('legacy Settings rejects a selected profile override after that exact target is replaced', async () => {
  const override = { mode: 'remote', baseUrl: 'https://override.example', token: 'override-token' }
  const replacement = { ...override, token: 'replacement-token' }

  await assert.rejects(
    resolveLegacyApiConnection(
      {
        legacyConnection: override,
        legacyConnectionProfile: 'worker',
        profile: 'worker',
        path: '/api/config'
      },
      'worker',
      async () => replacement
    ),
    /Backend changed/
  )
})

test('registered writes fail closed when the same id resolves to a replaced descriptor', async () => {
  const original = {
    mode: 'remote',
    baseUrl: 'https://original.example',
    token: 'original-token',
    headers: { 'Cf-Access-Client-Id': 'original-client' },
    remoteKind: 'ssh',
    remoteHost: 'operator@original-host'
  }

  for (const replacement of [
    { ...original, baseUrl: 'https://replacement.example' },
    { ...original, token: 'replacement-token' },
    { ...original, headers: { 'Cf-Access-Client-Id': 'replacement-client' } },
    { ...original, remoteHost: 'operator@replacement-host' }
  ]) {
    let release!: () => void

    const pending = new Promise<typeof replacement>(resolve => {
      release = () => resolve(replacement)
    })

    const request = { connectionId: 'same-id', connectionOwner: original, method: 'PUT', path: '/api/config' }
    const write = resolveRegistryApiConnection(request, 'same-id', () => pending)

    release()
    await assert.rejects(write, /Backend changed/)
  }
})

test('all row-derived session mutation requests reject a same-ID descriptor replacement before reaching it', async () => {
  const listed = { mode: 'remote', baseUrl: 'https://listed.example', token: 'listed-token' }
  const replacement = { ...listed, baseUrl: 'https://replacement.example' }

  const [row] = bindSessionRowsToRoute(
    [{ _lineage_root_id: 'session-root', id: 'session-live', profile: 'default' }],
    listed
  )

  const mutateReplacement = vi.fn()

  const route = {
    connectionId: 'same-id',
    profile: row.profile,
    sessionRouteBinding: row._desktop_route_binding,
    sessionRouteId: row.id
  }

  for (const mutation of mutationContract) {
    const request = {
      ...route,
      ...(mutation.body ? { body: mutation.body } : {}),
      method: mutation.method,
      path: `/api/sessions/${mutation.targetId}`
    }

    await assert.rejects(
      resolveRegistryApiConnection(request, 'same-id', async () => replacement).then(mutateReplacement),
      /Backend changed/,
      mutation.action
    )
  }

  assert.equal(mutateReplacement.mock.calls.length, 0)
})

test('all legacy row mutation requests reject route replacement, including lineage-root pinning', async () => {
  const listed = { mode: 'remote', baseUrl: 'https://legacy-a.example', token: 'listed-token' }
  const replacement = { ...listed, baseUrl: 'https://legacy-b.example' }

  const [row] = bindSessionRowsToRoute(
    [{ _lineage_root_id: 'session-root', id: 'session-live', profile: 'default' }],
    listed
  )

  const dispatch = vi.fn()

  for (const mutation of mutationContract) {
    const request = {
      ...(mutation.body ? { body: mutation.body } : {}),
      method: mutation.method,
      path: `/api/sessions/${mutation.targetId}`,
      profile: row.profile,
      sessionRouteBinding: row._desktop_route_binding,
      sessionRouteId: row.id
    }

    await assert.rejects(
      dispatchLegacySessionRequest(request, row.profile, async () => replacement, dispatch),
      /Backend changed/,
      mutation.action
    )
  }

  assert.equal(dispatch.mock.calls.length, 0)
})

test('registered session route binding rejects refreshed credentials for the same address', async () => {
  const listed = {
    mode: 'remote',
    baseUrl: 'https://listed.example',
    authMode: 'token',
    token: 'old-token',
    headers: { Authorization: 'old' }
  }

  const refreshed = { ...listed, token: 'new-token', headers: { Authorization: 'new' } }

  const request = {
    connectionId: 'same-id',
    method: 'DELETE',
    path: '/api/sessions/session-a',
    profile: 'default',
    sessionRouteBinding: createSessionRouteBinding('session-a', 'default', listed)
  }

  await assert.rejects(
    resolveRegistryApiConnection(request, 'same-id', async () => refreshed),
    /Backend changed/
  )
})

test('registered cross-profile writes validate the captured source owner around target acquisition', async () => {
  const source = { baseUrl: 'http://127.0.0.1:9001', mode: 'local', token: 'source-token' }
  const target = { baseUrl: 'http://127.0.0.1:9002', mode: 'local', token: 'target-token' }
  const ensureBackend = vi.fn(async (_connectionId, profile) => (profile === 'research' ? target : source))

  const request = {
    connectionId: 'same-id',
    connectionOwner: source,
    connectionOwnerProfile: 'default',
    method: 'DELETE',
    path: '/api/sessions/session-a',
    profile: 'research'
  }

  assert.equal(await resolveRegistryApiConnection(request, 'same-id', ensureBackend), target)
  assert.deepEqual(ensureBackend.mock.calls, [
    ['same-id', 'default', undefined],
    ['same-id', 'research', undefined],
    ['same-id', 'default', undefined]
  ])
})

test('registered cross-profile writes reject source replacement after target acquisition', async () => {
  const source = { baseUrl: 'http://127.0.0.1:9001', mode: 'local', token: 'source-token' }
  const target = { baseUrl: 'http://127.0.0.1:9002', mode: 'local', token: 'target-token' }
  const replacement = { ...source, token: 'replacement-token' }
  let sourceReads = 0

  const ensureBackend = vi.fn(async (_connectionId, profile) => {
    if (profile === 'research') {
      return target
    }

    sourceReads += 1

    return sourceReads === 1 ? source : replacement
  })

  await assert.rejects(
    resolveRegistryApiConnection(
      {
        connectionId: 'same-id',
        connectionOwner: source,
        connectionOwnerProfile: 'default',
        method: 'DELETE',
        path: '/api/sessions/session-a',
        profile: 'research'
      },
      'same-id',
      ensureBackend
    ),
    /Backend changed/
  )
})

test('legacy session interception preserves global owner validation while dispatching a profile override', async () => {
  const original = {
    mode: 'remote',
    baseUrl: 'https://original.example',
    token: 'original-token',
    headers: { 'Cf-Access-Client-Id': 'original-client' }
  }

  const override = { ...original, baseUrl: 'https://worker.example', token: 'worker-token' }
  const ensureBackend = vi.fn(async profile => (profile === 'default' ? original : override))
  const dispatch = vi.fn(async (_connection?: typeof override) => override)

  const request = {
    legacyConnection: original,
    legacyConnectionProfile: 'default',
    method: 'DELETE',
    path: '/api/sessions/session-a',
    profile: 'worker',
    sessionRouteBinding: createSessionRouteBinding('session-a', 'worker', override)
  }

  assert.equal(await dispatchLegacySessionRequest(request, 'worker', ensureBackend, dispatch), override)
  assert.deepEqual(ensureBackend.mock.calls, [
    ['default', { passive: undefined }],
    ['worker', {}],
    ['default', { passive: undefined }]
  ])
  assert.equal(dispatch.mock.calls.length, 1)
  assert.equal(dispatch.mock.calls[0][0], override)
})

test('legacy session interception rejects a target route changed since the row was listed', async () => {
  const source = { mode: 'remote', baseUrl: 'https://source.example', token: 'source-token' }
  const listedTarget = { mode: 'remote', baseUrl: 'https://worker-one.example', token: 'old-token' }
  const replacementTarget = { mode: 'remote', baseUrl: 'https://worker-two.example', token: 'new-token' }
  const ensureBackend = vi.fn(async profile => (profile === 'default' ? source : replacementTarget))
  const dispatch = vi.fn()

  await assert.rejects(
    dispatchLegacySessionRequest(
      {
        legacyConnection: source,
        legacyConnectionProfile: 'default',
        method: 'DELETE',
        path: '/api/sessions/session-a',
        profile: 'worker',
        sessionRouteBinding: createSessionRouteBinding('session-a', 'worker', listedTarget)
      },
      'worker',
      ensureBackend,
      dispatch
    ),
    /Backend changed/
  )
  assert.equal(dispatch.mock.calls.length, 0)
})

test('legacy session route binding rejects refreshed credentials for the same address', async () => {
  const source = { mode: 'remote', baseUrl: 'https://source.example', token: 'source-token' }

  const listedTarget = {
    mode: 'remote',
    baseUrl: 'https://worker.example',
    token: 'old-token',
    headers: { Authorization: 'old' }
  }

  const currentTarget = { ...listedTarget, token: 'new-token', headers: { Authorization: 'new' } }
  const ensureBackend = vi.fn(async profile => (profile === 'default' ? source : currentTarget))
  const dispatch = vi.fn(async connection => connection)

  await assert.rejects(
    dispatchLegacySessionRequest(
      {
        legacyConnection: source,
        legacyConnectionProfile: 'default',
        method: 'DELETE',
        path: '/api/sessions/session-a',
        profile: 'worker',
        sessionRouteBinding: createSessionRouteBinding('session-a', 'worker', listedTarget)
      },
      'worker',
      ensureBackend,
      dispatch
    ),
    /Backend changed/
  )
  assert.equal(dispatch.mock.calls.length, 0)
})

test('session-list route bindings are opaque and row-specific', () => {
  const connection = {
    mode: 'remote',
    baseUrl: 'https://secret-route.example',
    token: 'distinctive-token',
    headers: { Authorization: 'distinctive-header' }
  }

  const rows = bindSessionRowsToRoute(
    [
      { id: 'session-a', profile: 'worker' },
      { id: 'session-b', profile: 'worker' }
    ],
    connection
  )

  const serialized = JSON.stringify(rows)

  assert.ok(rows[0]._desktop_route_binding)
  assert.notEqual(rows[0]._desktop_route_binding, rows[1]._desktop_route_binding)
  assert.equal(serialized.includes(connection.baseUrl), false)
  assert.equal(serialized.includes(connection.token), false)
  assert.equal(serialized.includes(connection.headers.Authorization), false)
})

test('legacy session interception rejects a stale captured owner before dispatch', async () => {
  const original = {
    mode: 'remote',
    baseUrl: 'https://original.example',
    token: 'original-token',
    headers: { 'Cf-Access-Client-Id': 'original-client' }
  }

  const replacement = { ...original, token: 'replacement-token' }
  const ensureBackend = vi.fn(async () => replacement)
  const dispatch = vi.fn(async () => replacement)

  await assert.rejects(
    dispatchLegacySessionRequest(
      {
        legacyConnection: original,
        legacyConnectionProfile: 'default',
        method: 'DELETE',
        path: '/api/sessions/session-a',
        profile: 'worker'
      },
      'worker',
      ensureBackend,
      dispatch
    ),
    /Backend changed/
  )
  assert.deepEqual(ensureBackend.mock.calls[0], ['default', { passive: undefined }])
  assert.equal(dispatch.mock.calls.length, 0)
})

test('legacy session interception rejects source replacement during target acquisition', async () => {
  const original = {
    mode: 'remote',
    baseUrl: 'https://original.example',
    token: 'original-token',
    headers: { 'Cf-Access-Client-Id': 'original-client' }
  }

  const replacement = { ...original, token: 'replacement-token' }
  const target = { ...original, baseUrl: 'https://worker.example', token: 'worker-token' }
  let source = original
  let releaseTarget!: () => void
  const targetPending = new Promise<void>(resolve => (releaseTarget = resolve))

  const ensureBackend = vi.fn(async profile => {
    if (profile === 'worker') {
      await targetPending

      return target
    }

    return source
  })

  const dispatch = vi.fn(async () => target)

  const request = {
    legacyConnection: original,
    legacyConnectionProfile: 'default',
    method: 'DELETE',
    path: '/api/sessions/session-a',
    profile: 'worker'
  }

  const pending = dispatchLegacySessionRequest(request, 'worker', ensureBackend, dispatch)
  await vi.waitFor(() => assert.equal(ensureBackend.mock.calls.length, 2))
  source = replacement
  releaseTarget()

  await assert.rejects(pending, /Backend changed/)
  assert.equal(dispatch.mock.calls.length, 0)
})

test('legacy session interception revalidates the source immediately before dispatch', async () => {
  const original = {
    mode: 'remote',
    baseUrl: 'https://original.example',
    token: 'original-token',
    headers: { 'Cf-Access-Client-Id': 'original-client' }
  }

  const replacement = { ...original, token: 'replacement-token' }
  const target = { ...original, baseUrl: 'https://worker.example', token: 'worker-token' }
  let sourceReads = 0

  const ensureBackend = vi.fn(async profile => {
    if (profile === 'worker') {
      return target
    }

    sourceReads += 1

    return sourceReads === 1 ? original : replacement
  })

  const dispatch = vi.fn(async () => target)

  await assert.rejects(
    dispatchLegacySessionRequest(
      {
        legacyConnection: original,
        legacyConnectionProfile: 'default',
        method: 'DELETE',
        path: '/api/sessions/session-a',
        profile: 'worker'
      },
      'worker',
      ensureBackend,
      dispatch
    ),
    /Backend changed/
  )
  assert.equal(dispatch.mock.calls.length, 0)
})

test('legacy session interception remains byte-compatible without a captured owner', async () => {
  const ensureBackend = vi.fn()
  const dispatch = vi.fn(async (_connection?: unknown) => ({ ok: true }))

  assert.deepEqual(
    await dispatchLegacySessionRequest(
      { method: 'DELETE', path: '/api/sessions/session-a', profile: 'worker' },
      'worker',
      ensureBackend,
      dispatch
    ),
    { ok: true }
  )
  assert.equal(ensureBackend.mock.calls.length, 0)
  assert.equal(dispatch.mock.calls.length, 1)
  assert.equal(dispatch.mock.calls[0][0], undefined)
})
