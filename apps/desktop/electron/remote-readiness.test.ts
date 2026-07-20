import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  checkRemoteHermesOnce,
  preparePooledRemoteBackend,
  preparePrimaryRemoteBackend,
  waitForRemoteHermes
} from './remote-readiness'

const remote = {
  baseUrl: 'https://gateway.example',
  mode: 'remote',
  source: 'settings',
  authMode: 'token',
  token: 'saved-token',
  wsUrl: 'wss://gateway.example/api/ws?token=saved-token'
}

for (const [seam, prepare] of [
  ['primary remote boot', preparePrimaryRemoteBackend],
  ['pooled remote boot', preparePooledRemoteBackend]
] as const) {
  test(`${seam} rejects HTTP-ready gateway when authenticated WebSocket fails`, async () => {
    let statusChecks = 0
    let wsChecks = 0
    let clock = 0

    await assert.rejects(
      prepare(remote, {
        fetchStatus: async () => {
          statusChecks += 1
        },
        probeWebSocket: async url => {
          wsChecks += 1
          assert.equal(url, 'wss://gateway.example/api/ws?token=saved-token')

          return { ok: false, reason: '401 rejected' }
        },
        now: () => clock,
        sleep: async ms => {
          clock += ms
        },
        readyTimeoutMs: 1,
        retryDelayMs: 1
      }),
      /saved token could not open the live \/api\/ws chat connection.*401 rejected.*Refresh the token/
    )

    assert.equal(statusChecks, 1)
    assert.equal(wsChecks, 1)
  })
}

test('token readiness checks status and returns the authenticated WebSocket URL', async () => {
  const calls: unknown[][] = []

  const ready = await checkRemoteHermesOnce(remote, {
    fetchStatus: async (baseUrl, token, options) => {
      calls.push(['status', baseUrl, token, options.timeoutMs])
    },
    probeWebSocket: async url => {
      calls.push(['ws', url])

      return { ok: true }
    }
  })

  assert.deepEqual(calls, [
    ['status', 'https://gateway.example', 'saved-token', 8_000],
    ['ws', 'wss://gateway.example/api/ws?token=saved-token']
  ])
  assert.equal(ready.wsUrl, 'wss://gateway.example/api/ws?token=saved-token')
})

test('OAuth readiness uses a fresh single-use ticket for the renderer after probing', async () => {
  const calls: string[] = []
  let ticket = 0

  const ready = await checkRemoteHermesOnce(
    { baseUrl: 'https://gateway.example/prefix', authMode: 'oauth' },
    {
      fetchStatus: async (_baseUrl, token) => {
        assert.equal(token, null)
        calls.push('status')
      },
      mintTicket: async () => {
        calls.push('ticket')

        return `ticket-${++ticket}`
      },
      probeWebSocket: async url => {
        calls.push('ws')
        assert.equal(url, 'wss://gateway.example/prefix/api/ws?ticket=ticket-1')

        return { ok: true }
      }
    }
  )

  assert.deepEqual(calls, ['status', 'ticket', 'ws', 'ticket'])
  assert.equal(ready.wsUrl, 'wss://gateway.example/prefix/api/ws?ticket=ticket-2')
})

test('remote readiness retries until both status and WebSocket are ready', async () => {
  let attempts = 0
  let clock = 0

  await waitForRemoteHermes(remote, {
    fetchStatus: async () => undefined,
    probeWebSocket: async () => {
      attempts += 1

      return attempts === 1 ? { ok: false, reason: 'booting' } : { ok: true }
    },
    now: () => clock,
    sleep: async ms => {
      clock += ms
    },
    readyTimeoutMs: 5,
    retryDelayMs: 1
  })

  assert.equal(attempts, 2)
})
