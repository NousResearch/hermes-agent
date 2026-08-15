import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  DEFAULT_HEALTH_PROBE_TIMEOUT_MS,
  isAuthRejectionError,
  isGatedMissingHealthError,
  isMissingHealthEndpointError,
  isReauthRequiredError,
  waitForHermesReady
} from './backend-health'

const GATE_401 = '401: {"error":"unauthenticated","detail":"Unauthorized","reason":"no_cookie","login_url":"/login"}'

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

test('probes health on a short timeout but leaves the legacy fallback its own', async () => {
  const timeouts: (number | undefined)[] = []

  await waitForHermesReady('http://127.0.0.1:9000', {
    fetchPublicJson: async (_url, options) => {
      timeouts.push(options?.timeoutMs)

      throw new Error('404: {"detail":"Not Found"}')
    },
    fetchJson: async (_url, _token, options) => {
      timeouts.push(options?.timeoutMs)

      return { version: 'old' }
    },
    sleep: async () => {},
    timeoutMs: 100,
    pollMs: 1
  })

  assert.deepEqual(timeouts, [DEFAULT_HEALTH_PROBE_TIMEOUT_MS, undefined])
})

test('aborts as superseded when the bootstrap signal fires', async () => {
  const controller = new AbortController()
  controller.abort()

  await assert.rejects(
    waitForHermesReady('http://127.0.0.1:9000', {
      signal: controller.signal,
      fetchPublicJson: async () => {
        throw new Error('should not probe after abort')
      },
      fetchJson: async () => {
        throw new Error('should not probe after abort')
      },
      timeoutMs: 100,
      pollMs: 1
    }),
    (error: any) => error.kind === 'superseded'
  )
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

// --- Gated backends that predate /api/health (release 0.19.0 and earlier) ---
//
// The dashboard auth gate runs ahead of the SPA catch-all, so on a backend
// without the route an ANONYMOUS probe is rejected as unauthenticated rather
// than 404 — verified against a simulated 0.19.0 backend:
//   credential-free: /api/health -> 401 no_cookie, /api/status -> 200
//   credentialed:    /api/health -> 404,           /api/sessions -> 200

test('anonymous gate-shaped 401 falls back to /api/status (backend predates /api/health)', async () => {
  const calls: string[][] = []

  await waitForHermesReady('http://192.168.1.132:9119', {
    token: null,
    fetchPublicJson: async url => {
      calls.push(['public', url])
      throw new Error(GATE_401)
    },
    fetchJson: async (url, token) => {
      calls.push(['token', url, token == null ? 'null' : token])

      return { version: '0.19.0', auth_required: true }
    },
    sleep: async () => {},
    timeoutMs: 100,
    pollMs: 1
  })

  assert.deepEqual(calls, [
    ['public', 'http://192.168.1.132:9119/api/health'],
    ['token', 'http://192.168.1.132:9119/api/status', 'null']
  ])
})

test('a credentialed 401 fails fast for reauth instead of reporting a dead session ready', async () => {
  // The regression a blanket 401->fallback introduces: /api/status is public,
  // so an expired session would answer 200 and boot would report "ready",
  // deferring the no_cookie to the first real API call.
  const calls: string[][] = []

  await assert.rejects(
    waitForHermesReady('https://gateway.example', {
      token: 'session-token',
      fetchPublicJson: async () => {
        throw new Error('public probe must not be used when credentialed')
      },
      fetchJson: async url => {
        calls.push(['status', url])

        return { version: '0.19.0' }
      },
      probeHealth: async url => {
        calls.push(['probe', url])
        throw new Error(GATE_401)
      },
      probeIsCredentialed: true,
      sleep: async () => {},
      timeoutMs: 100,
      pollMs: 1
    }),
    (error: any) => {
      assert.equal(isReauthRequiredError(error), true)
      assert.equal(error.needsOauthLogin, true)
      assert.match(error.message, /remote gateway session has expired/i)

      return true
    }
  )

  // Fail fast: never reached the public /api/status leg.
  assert.deepEqual(calls, [['probe', 'https://gateway.example/api/health']])
})

test('a credentialed 403 is also a terminal reauth failure', async () => {
  await assert.rejects(
    waitForHermesReady('https://gateway.example', {
      fetchPublicJson: async () => ({}),
      fetchJson: async () => ({}),
      probeHealth: async () => {
        throw new Error('403: {"detail":"Forbidden"}')
      },
      probeIsCredentialed: true,
      sleep: async () => {},
      timeoutMs: 100,
      pollMs: 1
    }),
    (error: any) => isReauthRequiredError(error)
  )
})

test('a credentialed probe still uses the 404 fallback for a genuinely missing route', async () => {
  // With credentials the gate lets the request through to the SPA catch-all,
  // so an old backend answers a real 404 — that must still fall back, not be
  // mistaken for a rejected session.
  const calls: string[][] = []

  await waitForHermesReady('https://gateway.example', {
    token: 'session-token',
    fetchPublicJson: async () => {
      throw new Error('public probe must not be used when credentialed')
    },
    fetchJson: async url => {
      calls.push(['status', url])

      return { version: '0.19.0' }
    },
    probeHealth: async url => {
      calls.push(['probe', url])
      throw new Error('404: {"detail":"Not Found"}')
    },
    probeIsCredentialed: true,
    sleep: async () => {},
    timeoutMs: 100,
    pollMs: 1
  })

  assert.deepEqual(calls, [
    ['probe', 'https://gateway.example/api/health'],
    ['status', 'https://gateway.example/api/status']
  ])
})

test('a non-gate 401 keeps polling rather than skipping a misconfigured health route', async () => {
  const calls: string[][] = []
  let currentTime = 0

  await assert.rejects(
    waitForHermesReady('http://127.0.0.1:9000', {
      fetchPublicJson: async url => {
        calls.push(['public', url])
        throw new Error('401: {"detail":"Unauthorized"}')
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
    /401: \{"detail":"Unauthorized"\}/
  )

  assert.ok(calls.length > 0)
  assert.ok(calls.every(call => call[0] === 'public' && call[1].endsWith('/api/health')))
})

test('credentialed 5xx and 429 keep polling — only 401/403 are terminal', async () => {
  for (const transient of ['500: boom', '429: {"detail":"Too Many Requests"}']) {
    let attempts = 0
    let currentTime = 0

    await assert.rejects(
      waitForHermesReady('https://gateway.example', {
        fetchPublicJson: async () => ({}),
        fetchJson: async () => ({}),
        probeHealth: async () => {
          attempts += 1
          throw new Error(transient)
        },
        probeIsCredentialed: true,
        sleep: async () => {},
        now: () => {
          currentTime += 20

          return currentTime
        },
        timeoutMs: 100,
        pollMs: 1
      }),
      (error: any) => isReauthRequiredError(error) === false
    )

    assert.ok(attempts > 1, `${transient} should have retried, got ${attempts} attempt(s)`)
  }
})

test('error-shape predicates', () => {
  assert.equal(isGatedMissingHealthError(new Error(GATE_401)), true)
  assert.equal(isGatedMissingHealthError(new Error('401: {"detail":"Unauthorized"}')), false)
  assert.equal(isGatedMissingHealthError(new Error('404: {"detail":"Not Found"}')), false)

  assert.equal(isAuthRejectionError(new Error(GATE_401)), true)
  assert.equal(isAuthRejectionError(new Error('403: {"detail":"Forbidden"}')), true)
  assert.equal(isAuthRejectionError(new Error('404: {"detail":"Not Found"}')), false)
  assert.equal(isAuthRejectionError(new Error('429: slow down')), false)
  assert.equal(isAuthRejectionError(new Error('500: boom')), false)

  // A gated 401 must NOT be conflated with a missing route by the 404 predicate.
  assert.equal(isMissingHealthEndpointError(new Error(GATE_401)), false)
})
