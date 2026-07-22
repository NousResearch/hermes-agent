import assert from 'node:assert/strict'
import { test } from 'vitest'

import {
  HERMES_HEALTH_REQUEST_TIMEOUT_MS,
  HERMES_READY_RETRY_MS,
  HERMES_READY_TIMEOUT_MS,
  isMissingHealthEndpointError,
  isReadyHealthResponse,
  waitForHermesReadiness
} from './backend-health'

test('uses the confirmed readiness timing defaults', () => {
  assert.equal(HERMES_HEALTH_REQUEST_TIMEOUT_MS, 2_500)
  assert.equal(HERMES_READY_RETRY_MS, 500)
  assert.equal(HERMES_READY_TIMEOUT_MS, 45_000)
})

test('recognizes only the expected health response as ready', () => {
  assert.equal(isReadyHealthResponse({ ok: true, status: 'ready' }), true)
  assert.equal(isReadyHealthResponse({ ok: true, status: 'starting' }), false)
  assert.equal(isReadyHealthResponse({ ok: false, status: 'ready' }), false)
  assert.equal(isReadyHealthResponse(null), false)
})

test('probes healthz first with a 2500ms request timeout', async () => {
  const calls: Array<{ url: string; timeoutMs: number }> = []

  await waitForHermesReadiness('http://127.0.0.1:8000', async (url, options) => {
    calls.push({ url, timeoutMs: options.timeoutMs })

    return { ok: true, status: 'ready' }
  })

  assert.deepEqual(calls, [{ url: 'http://127.0.0.1:8000/api/healthz', timeoutMs: 2_500 }])
})

test('permanently falls back to status when healthz returns 404', async () => {
  const paths: string[] = []

  await waitForHermesReadiness(
    'https://remote.example',
    async url => {
      const path = new URL(url).pathname
      paths.push(path)

      if (paths.length === 1) {
        throw new Error('404: Not Found')
      }

      if (paths.length === 2) {
        throw new Error('connect ECONNRESET')
      }

      return { status: 'ok' }
    },
    { sleep: async () => {} }
  )

  assert.deepEqual(paths, ['/api/healthz', '/api/status', '/api/status'])
})

test('permanently falls back when the existing fetch helper reports an HTML endpoint miss', async () => {
  const paths: string[] = []

  await waitForHermesReadiness('https://remote.example', async url => {
    const path = new URL(url).pathname
    paths.push(path)

    if (path === '/api/healthz') {
      throw new Error(
        'Expected JSON from https://remote.example/api/healthz but got HTML (status 200). ' +
          'The endpoint is likely missing on the Hermes backend.'
      )
    }

    return { status: 'ok' }
  })

  assert.deepEqual(paths, ['/api/healthz', '/api/status'])
})

test('does not fall back for 5xx, timeout, network, or malformed readiness responses', async () => {
  const failures = [
    new Error('500: Internal Server Error'),
    new Error('Timed out connecting to Hermes backend after 2500ms'),
    new Error('connect ECONNREFUSED 127.0.0.1:8000'),
    null
  ]

  const paths: string[] = []

  await waitForHermesReadiness(
    'http://127.0.0.1:8000',
    async url => {
      paths.push(new URL(url).pathname)
      const failure = failures.shift()

      if (failure) {
        throw failure
      }

      if (failure === null) {
        return { ok: true, status: 'starting' }
      }

      return { ok: true, status: 'ready' }
    },
    { sleep: async () => {} }
  )

  assert.deepEqual(paths, [
    '/api/healthz',
    '/api/healthz',
    '/api/healthz',
    '/api/healthz',
    '/api/healthz'
  ])
})

test('classifies only explicit endpoint-missing errors for legacy fallback', () => {
  assert.equal(isMissingHealthEndpointError(new Error('404: Not Found')), true)
  assert.equal(
    isMissingHealthEndpointError(
      new Error(
        'Expected JSON from http://localhost/api/healthz but got HTML (status 200). ' +
          'The endpoint is likely missing on the Hermes backend.'
      )
    ),
    true
  )
  assert.equal(isMissingHealthEndpointError(new Error('500: Internal Server Error')), false)
  assert.equal(isMissingHealthEndpointError(new Error('Timed out connecting after 2500ms')), false)
  assert.equal(isMissingHealthEndpointError(new Error('Unexpected token < in JSON')), false)
})

test('retries every 500ms and stops at the 45s overall deadline', async () => {
  let nowMs = 0
  const requestTimeouts: number[] = []
  const sleeps: number[] = []

  await assert.rejects(
    waitForHermesReadiness(
      'http://127.0.0.1:8000',
      async (_url, options) => {
        requestTimeouts.push(options.timeoutMs)
        throw new Error('connect ECONNREFUSED')
      },
      {
        now: () => nowMs,
        sleep: async delayMs => {
          sleeps.push(delayMs)
          nowMs += delayMs
        }
      }
    ),
    /Hermes backend did not become ready: connect ECONNREFUSED/
  )

  assert.equal(nowMs, 45_000)
  assert.ok(sleeps.length > 1)
  assert.ok(sleeps.every(delayMs => delayMs === 500))
  assert.equal(requestTimeouts[0], 2_500)
  assert.equal(requestTimeouts.at(-1), 500)
})

test('does not issue a request when the deadline expires between loop checks', async () => {
  const times = [0, 44_999, 45_000]
  let requests = 0

  await assert.rejects(
    waitForHermesReadiness(
      'http://127.0.0.1:8000',
      async () => {
        requests += 1

        return { ok: true, status: 'ready' }
      },
      {
        now: () => times.shift() ?? 45_000,
        sleep: async () => {
          assert.fail('must not sleep after the overall deadline')
        }
      }
    ),
    /Hermes backend did not become ready: timeout/
  )

  assert.equal(requests, 0)
})

test('stops retrying immediately when an abort signal supersedes SSH bootstrap', async () => {
  const controller = new AbortController()
  let requests = 0

  await assert.rejects(
    waitForHermesReadiness(
      'http://127.0.0.1:8000',
      async () => {
        requests += 1
        throw new Error('connect ECONNREFUSED')
      },
      {
        signal: controller.signal,
        sleep: async () => {
          controller.abort()
        }
      }
    ),
    (error: any) => error?.kind === 'superseded'
  )

  assert.equal(requests, 1)
})
