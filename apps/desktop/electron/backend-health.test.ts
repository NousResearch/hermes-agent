import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  DEFAULT_HEALTH_PROBE_TIMEOUT_MS,
  isAuthRejectionError,
  isGatedMissingHealthError,
  isMissingHealthEndpointError,
  isNousCloudAgentUrl,
  isReauthRequiredError,
  isServerSideHttpError,
  makeNousCloudBackendDownError,
  makeUnsignedOauthError,
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

test('unsigned OAuth is a terminal reauth failure; needsOauthLogin alone is not', () => {
  // The unsigned-in throw must set isReauthRequired so startHermes latches.
  // needsOauthLogin alone (ticket 401/403) stays a Sign-in hint, not a latch —
  // a lapsed AT cookie can still rotate from a live RT on the next mint.
  const unsigned = makeUnsignedOauthError() as any

  assert.equal(unsigned.needsOauthLogin, true)
  assert.equal(unsigned.isReauthRequired, true)
  assert.equal(isReauthRequiredError(unsigned), true)
  assert.match(unsigned.message, /not signed in/i)
  assert.equal(isReauthRequiredError({ needsOauthLogin: true }), false)
  assert.equal(isReauthRequiredError(new Error('Could not reach the remote Hermes gateway')), false)
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

test('isServerSideHttpError detects 502/503/504', () => {
  // 503 — server-side fault
  const result503 = isServerSideHttpError(new Error('503: Service Unavailable'))
  assert.ok(result503, 'should detect 503')
  assert.equal(result503?.statusCode, 503)
  assert.equal(result503?.detail, '503: Service Unavailable')

  // 502
  const result502 = isServerSideHttpError(new Error('502: Bad Gateway'))
  assert.ok(result502, 'should detect 502')
  assert.equal(result502?.statusCode, 502)

  // 504
  const result504 = isServerSideHttpError(new Error('504: Gateway Timeout'))
  assert.ok(result504, 'should detect 504')
  assert.equal(result504?.statusCode, 504)

  // 500 is NOT a server-side HTTP error per our definition (keeps polling)
  const result500 = isServerSideHttpError(new Error('500: Internal Server Error'))
  assert.equal(result500, null)

  // 401/403/404/429 are not server-side faults
  assert.equal(isServerSideHttpError(new Error('401: Unauthorized')), null)
  assert.equal(isServerSideHttpError(new Error('403: Forbidden')), null)
  assert.equal(isServerSideHttpError(new Error('404: Not Found')), null)
  assert.equal(isServerSideHttpError(new Error('429: Too Many Requests')), null)

  // Non-HTTP errors (timeouts, network failures) don't match the pattern
  assert.equal(isServerSideHttpError(new Error('connect ECONNREFUSED')), null)
  assert.equal(isServerSideHttpError(null), null)
  assert.equal(isServerSideHttpError('503: something'), null) // not an Error
})

test('isNousCloudAgentUrl detects cloud agent hosts', () => {
  // Positive cases
  assert.equal(isNousCloudAgentUrl('https://ares-3009.agents.nousresearch.com'), true)
  assert.equal(isNousCloudAgentUrl('https://ares-3009.agents.nousresearch.com/api/health'), true)
  assert.equal(isNousCloudAgentUrl('http://test.agents.nousresearch.com'), true)

  // Negative cases
  assert.equal(isNousCloudAgentUrl('http://127.0.0.1:9000'), false)
  assert.equal(isNousCloudAgentUrl('https://gateway.example.com'), false)
  assert.equal(isNousCloudAgentUrl('https://nousresearch.com'), false)
  assert.equal(isNousCloudAgentUrl('not-a-url'), false)
})

test('waitForHermesReady surfaces actionable error for cloud agent 503', async () => {
  let attempts = 0
  const currentTime = { value: 0 }

  try {
    await waitForHermesReady('https://ares-3009.agents.nousresearch.com', {
      fetchPublicJson: async () => {
        attempts++
        // Always return 503
        throw new Error('503: Service Unavailable')
      },
      fetchJson: async () => {
        throw new Error('503: Service Unavailable')
      },
      sleep: async () => {},
      // Advance the mock clock per poll — a frozen now() never crosses the
      // deadline and the readiness loop spins forever (hung the whole vitest
      // electron project for 20m in CI).
      now: () => {
        currentTime.value += 20

        return currentTime.value
      },
      timeoutMs: 100,
      pollMs: 1
    })
    assert.fail('should have thrown')
  } catch (error: any) {
    assert.ok(error.message.includes('Nous Cloud agent'), `unexpected message: ${error.message}`)
    assert.ok(error.message.includes('503'), `should mention status code: ${error.message}`)
    assert.ok(error.message.includes('portal.nousresearch.com'), `should mention portal: ${error.message}`)
    assert.ok(error.message.includes('discord.gg/NousResearch'), `should mention Discord: ${error.message}`)
    assert.equal(error.isCloudBackendDown, true)
    assert.equal(error.statusCode, 503)
    assert.ok(attempts > 1, 'should have retried before failing')
  }
})

test('waitForHermesReady does not cloud-wrap non-cloud 503 errors', async () => {
  const currentTime = { value: 0 }

  try {
    await waitForHermesReady('http://127.0.0.1:9000', {
      fetchPublicJson: async () => {
        throw new Error('503: Service Unavailable')
      },
      fetchJson: async () => {
        throw new Error('503: Service Unavailable')
      },
      sleep: async () => {},
      // Same advancing clock as above — frozen now() = infinite loop.
      now: () => {
        currentTime.value += 20

        return currentTime.value
      },
      timeoutMs: 100,
      pollMs: 1
    })
    assert.fail('should have thrown')
  } catch (error: any) {
    // Non-cloud URLs get the generic message
    assert.ok(error.message.includes('did not become ready'), `unexpected message: ${error.message}`)
    assert.equal(error.isCloudBackendDown, undefined)
  }
})

test('isServerSideHttpError detects structured statusCode even when the message is opaque', () => {
  const err = new Error('upstream unavailable') as any
  err.statusCode = 503
  const result = isServerSideHttpError(err)
  assert.ok(result)
  assert.equal(result?.statusCode, 503)
  assert.equal(result?.detail, 'upstream unavailable')

  const err502 = new Error('bad gateway') as any
  err502.statusCode = 502
  assert.equal(isServerSideHttpError(err502)?.statusCode, 502)

  const err504 = new Error('gateway timeout') as any
  err504.statusCode = 504
  assert.equal(isServerSideHttpError(err504)?.statusCode, 504)
})

test('isServerSideHttpError rejects non-Error inputs even with a 503-shaped value', () => {
  // The structured path requires an actual Error (the fetch layer attaches
  // statusCode to an Error instance); a bare string/null/number must not be
  // misclassified by the legacy prefix fallback.
  assert.equal(isServerSideHttpError('503: something'), null)
  assert.equal(isServerSideHttpError({ statusCode: 503 }), null)
  assert.equal(isServerSideHttpError(null), null)
  assert.equal(isServerSideHttpError(503), null)
})

test('isServerSideHttpError structured path excludes 500/401/403/404/429 even when statusCode is attached', () => {
  for (const code of [500, 401, 403, 404, 429]) {
    const err = new Error(`HTTP ${code}`) as any
    err.statusCode = code
    assert.equal(isServerSideHttpError(err), null, `should reject statusCode ${code}`)
  }
})

test('makeNousCloudBackendDownError produces the Cloud shape and preserves cause', () => {
  const err = new Error('upstream unavailable') as any
  err.statusCode = 503
  const result = makeNousCloudBackendDownError('https://ares-3009.agents.nousresearch.com', err)
  assert.ok(result)
  assert.equal((result as any).isCloudBackendDown, true)
  assert.equal((result as any).statusCode, 503)
  assert.equal((result as any).cause, err)
  assert.ok(result?.message.includes('Nous Cloud agent ares-3009.agents.nousresearch.com is down'))
})

test('makeNousCloudBackendDownError returns null for a Cloud 401 (routes to reauth)', () => {
  const err = new Error('Unauthorized') as any
  err.statusCode = 401
  assert.equal(makeNousCloudBackendDownError('https://ares-3009.agents.nousresearch.com', err), null)
})

test('makeNousCloudBackendDownError returns null for a non-Cloud 503 (generic remote failure)', () => {
  const err = new Error('Service Unavailable') as any
  err.statusCode = 503
  assert.equal(makeNousCloudBackendDownError('https://gateway.example.com', err), null)
  assert.equal(makeNousCloudBackendDownError('http://127.0.0.1:9000', err), null)
})

test('makeNousCloudBackendDownError preserves legacy string-prefix compatibility', () => {
  const result = makeNousCloudBackendDownError(
    'https://ares-3009.agents.nousresearch.com',
    new Error('503: Service Unavailable')
  )

  assert.ok(result)
  assert.equal((result as any).isCloudBackendDown, true)
  assert.equal((result as any).statusCode, 503)
})
