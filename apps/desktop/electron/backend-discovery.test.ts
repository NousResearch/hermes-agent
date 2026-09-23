import assert from 'node:assert/strict'

import { test } from 'vitest'

import { attachToHostBackend } from './host-backend-attach'
import { parseHostBackendRendezvous, validateHostBackendIdentity } from './host-backend-rendezvous'
import { runPrimaryBackendStartup } from './primary-backend-startup'

const LEDGER = JSON.stringify([
  {
    argv: 'hermes serve --host 127.0.0.1 --port 0',
    create_time: 1_000,
    host: '127.0.0.1',
    install: 'abc',
    pid: 4711,
    port: 65_238,
    profile: 'ops',
    purpose: 'serve',
    registered_at: 2_000
  }
])

function attachDeps(ledger: string | null) {
  return {
    log: () => {},
    probeWebSocket: async () => ({ ok: true }),
    readLedger: () => ledger,
    resolveServedToken: async () => 'served-token',
    waitForReady: async () => undefined
  }
}

/**
 * Multiplex-only invariant: a backend already running on the HOST is THE
 * backend — even one registered by another profile. Startup attaches to it and
 * spawns nothing.
 */
test('a running backend record makes startup attach and spawn zero processes', async () => {
  let spawns = 0

  const setup = await runPrimaryBackendStartup({
    assertCurrentAttempt: () => {},
    attachHostBackend: () => attachToHostBackend({ isolated: false, ledgerPath: '/ledger.json' }, attachDeps(LEDGER)),
    connectRemote: async () => ({ mode: 'remote' }),
    ensureLocalRuntime: async backend => backend,
    prepareLocalBackend: () => {
      spawns += 1

      return { label: 'spawned' }
    },
    resolveRemote: async () => null,
    waitForDecision: async () => 'continue-local' as const,
    waitForLocalStart: async () => undefined
  })

  assert.equal(setup.kind, 'attached')
  assert.equal(spawns, 0, 'startup must not prepare/spawn a backend when the host already has one')
  assert.deepEqual(setup.kind === 'attached' ? setup.attached : null, {
    baseUrl: 'http://127.0.0.1:65238',
    pid: 4711,
    port: 65238,
    token: 'served-token',
    wsUrl: 'ws://127.0.0.1:65238/api/ws?token=served-token'
  })
})

/** The only case that may start a process: the host has no backend. */
test('no backend record spawns exactly one backend', async () => {
  let spawns = 0

  const setup = await runPrimaryBackendStartup({
    assertCurrentAttempt: () => {},
    attachHostBackend: () => attachToHostBackend({ isolated: false, ledgerPath: '/ledger.json' }, attachDeps(null)),
    connectRemote: async () => ({ mode: 'remote' }),
    ensureLocalRuntime: async backend => backend,
    prepareLocalBackend: () => {
      spawns += 1

      return { label: 'spawned' }
    },
    resolveRemote: async () => null,
    waitForDecision: async () => 'continue-local' as const,
    waitForLocalStart: async () => undefined
  })

  assert.equal(setup.kind, 'local')
  assert.equal(spawns, 1)
})

/** A record that fails validation is not a backend: fall through to spawning. */
test('a record whose backend rejects the session token does not attach', async () => {
  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/ledger.json' },
    { ...attachDeps(LEDGER), probeWebSocket: async () => ({ ok: false, reason: 'unauthorized' }) }
  )

  assert.equal(attached, null)
})

test('a private host rendezvous token can attach when the dashboard html withholds it', async () => {
  const token = 'private-rendezvous-token'

  const rendezvous = parseHostBackendRendezvous(
    JSON.stringify({
      role: 'serve',
      pid: 4711,
      port: 65_238,
      host: '127.0.0.1',
      createTime: 1_000,
      protocolVersion: 1,
      tokenFingerprint: '6b0820bb6bbc5593',
      profiles: ['ops'],
      updatedAt: '2026-09-21T18:00:00.000Z'
    }),
    token
  )

  // The fingerprint must match the private token before the candidate can be
  // considered. This keeps the fixture honest if the token changes later.
  assert.ok(rendezvous)
  assert.equal(
    parseHostBackendRendezvous(
      JSON.stringify({
        role: 'serve',
        protocolVersion: 1,
        pid: 4711,
        port: 65_238,
        tokenFingerprint: 'wrong-fingerprint'
      }),
      token
    ),
    null
  )
  assert.equal(
    parseHostBackendRendezvous(JSON.stringify({ role: 'serve', protocolVersion: 1, pid: 4711, port: 65_238 }), token),
    null
  )
  assert.equal(
    parseHostBackendRendezvous(
      JSON.stringify({
        role: 'serve',
        protocolVersion: 1,
        pid: 4711,
        port: 65_238,
        createTime: 'bad',
        tokenFingerprint: '6b0820bb6bbc5593'
      }),
      token
    ),
    null
  )

  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/missing-ledger.json' },
    {
      ...attachDeps(null),
      resolveServedToken: async () => null,
      probeHostIdentity: async (_baseUrl, probeToken, record) => {
        assert.equal(probeToken, token)
        assert.equal(record.createTime, 1_000)

        return { ok: true }
      },
      readRendezvous: () => ({ candidate: rendezvous, port: rendezvous?.record.port ?? 65_238 }),
      waitForReady: async (_baseUrl, servedToken) => assert.equal(servedToken, token),
      probeWebSocket: async wsUrl => {
        assert.match(wsUrl, /token=private-rendezvous-token/)

        return { ok: true }
      }
    }
  )

  assert.equal(attached?.token, token)
})

test('host identity proof rejects malformed or headless owners', () => {
  const record = {
    createTime: 1_000,
    host: '127.0.0.1',
    pid: 4711,
    port: 65_238,
    profile: 'ops',
    purpose: 'serve',
    registeredAt: 2_000
  }

  assert.deepEqual(
    validateHostBackendIdentity(
      { ok: true, protocolVersion: 1, pid: 4711, role: 'serve', servesSpa: true, createTime: 1_000 },
      record
    ),
    { ok: true }
  )
  assert.equal(
    validateHostBackendIdentity(
      { ok: true, protocolVersion: 1, pid: 4711, role: 'serve', servesSpa: false, createTime: 1_000 },
      record
    ).ok,
    false
  )
  assert.equal(
    validateHostBackendIdentity({ ok: true, protocolVersion: 1, pid: 4711, role: 'serve', servesSpa: true }, record).ok,
    false
  )
})

test('a rejected private token retries with the served dashboard token', async () => {
  const rendezvous = parseHostBackendRendezvous(
    JSON.stringify({
      role: 'serve',
      pid: 4711,
      port: 65_238,
      host: '127.0.0.1',
      createTime: 1_000,
      protocolVersion: 1,
      tokenFingerprint: '6b0820bb6bbc5593',
      updatedAt: '2026-09-21T18:00:00.000Z'
    }),
    'private-rendezvous-token'
  )

  const identityTokens: string[] = []

  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/missing-ledger.json' },
    {
      ...attachDeps(null),
      resolveServedToken: async () => 'served-dashboard-token',
      probeHostIdentity: async (_baseUrl, token) => {
        identityTokens.push(token)

        return { ok: token === 'served-dashboard-token' }
      },
      readRendezvous: () => ({ candidate: rendezvous, port: rendezvous?.record.port ?? 65_238 }),
      waitForReady: async (_baseUrl, token) => assert.equal(token, 'served-dashboard-token'),
      probeWebSocket: async wsUrl => {
        assert.match(wsUrl, /token=served-dashboard-token/)

        return { ok: true }
      }
    }
  )

  assert.deepEqual(identityTokens, ['private-rendezvous-token', 'served-dashboard-token'])
  assert.equal(attached?.token, 'served-dashboard-token')
})

test('a failed rendezvous proof does not bypass the same endpoint through the ledger', async () => {
  const rendezvous = parseHostBackendRendezvous(
    JSON.stringify({
      role: 'serve',
      pid: 4711,
      port: 65_238,
      host: '127.0.0.1',
      createTime: 1_000,
      protocolVersion: 1,
      tokenFingerprint: '6b0820bb6bbc5593',
      updatedAt: '2026-09-21T18:00:00.000Z'
    }),
    'private-rendezvous-token'
  )

  let identityProbes = 0

  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/ledger.json' },
    {
      ...attachDeps(
        JSON.stringify([
          {
            ...JSON.parse(LEDGER)[0],
            pid: 9999
          }
        ])
      ),
      probeHostIdentity: async () => {
        identityProbes += 1

        return { ok: false, reason: 'stale record' }
      },
      readRendezvous: () => ({ candidate: rendezvous, port: rendezvous?.record.port ?? 65_238 })
    }
  )

  assert.equal(identityProbes, 2)
  assert.equal(attached, null)
})

test('a failed rendezvous proof still allows a different ledger endpoint', async () => {
  const rendezvous = parseHostBackendRendezvous(
    JSON.stringify({
      role: 'serve',
      pid: 4711,
      port: 65_238,
      host: '127.0.0.1',
      createTime: 1_000,
      protocolVersion: 1,
      tokenFingerprint: '6b0820bb6bbc5593',
      updatedAt: '2026-09-21T18:00:00.000Z'
    }),
    'private-rendezvous-token'
  )

  const differentLedger = JSON.stringify([
    {
      ...JSON.parse(LEDGER)[0],
      pid: 4811,
      port: 65_239
    }
  ])

  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/ledger.json' },
    {
      ...attachDeps(differentLedger),
      probeHostIdentity: async () => ({ ok: false, reason: 'stale record' }),
      readRendezvous: () => ({ candidate: rendezvous, port: rendezvous?.record.port ?? 65_238 })
    }
  )

  assert.equal(attached?.token, 'served-token')
})

test('an invalid rendezvous record blocks a same-port ledger fallback', async () => {
  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/ledger.json' },
    {
      ...attachDeps(LEDGER),
      readRendezvous: () => ({ candidate: null, port: 65_238 })
    }
  )

  assert.equal(attached, null)
})
