import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isMissingHealthEndpointError, waitForHermesReady } from './backend-health'

test('uses lightweight /api/health for current backends', async () => {
  const calls: string[][] = []

  await waitForHermesReady('http://127.0.0.1:9000/', {
    token: 'secret-token',
    fetchPublicJson: async url => {
      calls.push(['public', url])

      return { ok: true }
    },
    fetchJson: async url => {
      calls.push(['token', url])
      throw new Error('status should not be called')
    },
    sleep: async () => {},
    timeoutMs: 100,
    pollMs: 1
  })

  assert.deepEqual(calls, [['public', 'http://127.0.0.1:9000/api/health']])
})

test('falls back to /api/status only for old backends without /api/health', async () => {
  const calls: string[][] = []

  await waitForHermesReady('http://127.0.0.1:9000', {
    token: 'secret-token',
    fetchPublicJson: async url => {
      calls.push(['public', url])

      throw new Error('404: {"detail":"Not Found"}')
    },
    fetchJson: async (url, token) => {
      calls.push(['token', url, token ?? ''])

      return { version: 'old' }
    },
    sleep: async () => {},
    timeoutMs: 100,
    pollMs: 1
  })

  assert.deepEqual(calls, [
    ['public', 'http://127.0.0.1:9000/api/health'],
    ['token', 'http://127.0.0.1:9000/api/status', 'secret-token']
  ])
})

test('does not fall back to heavyweight /api/status for transient health failures', async () => {
  const calls: string[][] = []
  let currentTime = 0

  await assert.rejects(
    waitForHermesReady('http://127.0.0.1:9000', {
      fetchPublicJson: async url => {
        calls.push(['public', url])
        throw new Error('Timed out connecting to Hermes backend after 15000ms')
      },
      fetchJson: async url => {
        calls.push(['token', url])
      },
      sleep: async () => {},
      now: () => {
        currentTime += 20

        return currentTime
      },
      timeoutMs: 50,
      pollMs: 1
    }),
    /Timed out connecting/
  )

  assert.ok(calls.length > 0)
  assert.ok(calls.every(call => call[0] === 'public' && call[1].endsWith('/api/health')))
})

test('recognizes missing-route shapes only', () => {
  assert.equal(isMissingHealthEndpointError(new Error('404: {"detail":"Not Found"}')), true)
  assert.equal(
    isMissingHealthEndpointError(
      new Error('Expected JSON from /api/health but got HTML. The endpoint is likely missing on the Hermes backend.')
    ),
    true
  )
  assert.equal(isMissingHealthEndpointError(new Error('Timed out connecting to Hermes backend after 15000ms')), false)
  assert.equal(isMissingHealthEndpointError(new Error('500: boom')), false)
})
