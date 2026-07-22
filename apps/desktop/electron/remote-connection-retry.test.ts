import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  isRetryableRemoteConnectionError,
  remoteHttpStatusError,
  requiresOauthLogin,
  resolveReadyRemoteConnectionWithRetry,
  resolveRemoteConnectionWithRetry
} from './remote-connection-retry'

test('401 and 403 require OAuth login and are never retried', async () => {
  for (const statusCode of [401, 403]) {
    let attempts = 0

    await assert.rejects(
      () =>
        resolveRemoteConnectionWithRetry(
          async () => {
            attempts += 1
            throw Object.assign(new Error(`${statusCode} response`), { statusCode })
          },
          { maxAttempts: 5, sleep: async () => undefined }
        ),
      error => requiresOauthLogin(error)
    )

    assert.equal(attempts, 1)
  }
})

test('production HTTP errors preserve status for immediate auth classification', () => {
  const error = remoteHttpStatusError(403, 'forbidden')

  assert.equal(error.message, '403: forbidden')
  assert.equal(error.statusCode, 403)
  assert.equal(requiresOauthLogin(error), true)
  assert.equal(isRetryableRemoteConnectionError(error), false)
})

test('wrapped sign-in-required failures are never retried', async () => {
  let attempts = 0

  const authError = Object.assign(new Error('Sign in again'), {
    cause: Object.assign(new Error('forbidden'), { statusCode: 403 }),
    needsOauthLogin: true
  })

  await assert.rejects(
    () =>
      resolveRemoteConnectionWithRetry(
        async () => {
          attempts += 1
          throw authError
        },
        { sleep: async () => undefined }
      ),
    error => error === authError
  )

  assert.equal(attempts, 1)
})

test('deeply wrapped auth rejections are never retried', async () => {
  let attempts = 0

  const authError = Object.assign(new Error('connection resolution failed'), {
    cause: Object.assign(new Error('ticket mint failed'), {
      cause: Object.assign(new Error('forbidden'), { statusCode: 403 })
    })
  })

  await assert.rejects(
    () =>
      resolveRemoteConnectionWithRetry(
        async () => {
          attempts += 1
          throw authError
        },
        { maxAttempts: 5, sleep: async () => undefined }
      ),
    error => error === authError
  )

  assert.equal(requiresOauthLogin(authError), true)
  assert.equal(attempts, 1)
})

test('timeouts and 5xx failures use fresh resolver calls with exponential backoff', async () => {
  const errors = [
    new Error('Timed out connecting to Hermes backend after 8000ms'),
    Object.assign(new Error('503 upstream unavailable'), { statusCode: 503 })
  ]

  const sleeps: number[] = []
  let attempts = 0

  const result = await resolveRemoteConnectionWithRetry(
    async () => {
      attempts += 1
      const error = errors.shift()

      if (error) {
        throw error
      }

      return { ticket: `fresh-ticket-${attempts}` }
    },
    {
      initialDelayMs: 500,
      maxAttempts: 5,
      sleep: async delayMs => {
        sleeps.push(delayMs)
      }
    }
  )

  assert.deepEqual(result, { ticket: 'fresh-ticket-3' })
  assert.equal(attempts, 3)
  assert.deepEqual(sleeps, [500, 1_000])
})

test('token readiness failures retry the complete connection attempt', async () => {
  let readinessAttempts = 0
  let resolverAttempts = 0

  const result = await resolveReadyRemoteConnectionWithRetry(
    async () => {
      resolverAttempts += 1

      return { baseUrl: 'https://remote.example', token: 'static-token' }
    },
    async () => {
      readinessAttempts += 1

      if (readinessAttempts === 1) {
        throw Object.assign(new Error('connection refused'), { code: 'ECONNREFUSED' })
      }
    },
    { maxAttempts: 2, sleep: async () => undefined }
  )

  assert.deepEqual(result, { baseUrl: 'https://remote.example', token: 'static-token' })
  assert.equal(resolverAttempts, 2)
  assert.equal(readinessAttempts, 2)
})

test('OAuth readiness retries resolve a fresh single-use ticket', async () => {
  const ticketsSeenByReadiness: string[] = []
  let resolverAttempts = 0

  const result = await resolveReadyRemoteConnectionWithRetry(
    async () => {
      resolverAttempts += 1

      return { wsUrl: `wss://remote.example/ws?ticket=fresh-${resolverAttempts}` }
    },
    async connection => {
      ticketsSeenByReadiness.push(connection.wsUrl)

      if (ticketsSeenByReadiness.length === 1) {
        throw Object.assign(new Error('503 upstream unavailable'), { statusCode: 503 })
      }
    },
    { maxAttempts: 2, sleep: async () => undefined }
  )

  assert.deepEqual(ticketsSeenByReadiness, [
    'wss://remote.example/ws?ticket=fresh-1',
    'wss://remote.example/ws?ticket=fresh-2'
  ])
  assert.equal(result?.wsUrl, 'wss://remote.example/ws?ticket=fresh-2')
})

test('readiness authentication failures bypass retry', async () => {
  let readinessAttempts = 0
  let resolverAttempts = 0
  const authError = Object.assign(new Error('forbidden'), { statusCode: 403 })

  await assert.rejects(
    () =>
      resolveReadyRemoteConnectionWithRetry(
        async () => {
          resolverAttempts += 1

          return { baseUrl: 'https://remote.example' }
        },
        async () => {
          readinessAttempts += 1
          throw authError
        },
        { maxAttempts: 5, sleep: async () => undefined }
      ),
    error => error === authError
  )

  assert.equal(resolverAttempts, 1)
  assert.equal(readinessAttempts, 1)
})

test('SSH wrappers preserve transient and permanent classification', () => {
  const refused = Object.assign(new Error('remote ownership probe failed'), { kind: 'transient-transport-error' })

  const transientWrapper = Object.assign(new Error(refused.message, { cause: refused }), {
    isSshBootstrap: true,
    kind: refused.kind,
    sshError: refused.kind
  })

  const auth = Object.assign(new Error('Permission denied (publickey).'), { kind: 'auth-failed' })

  const permanentWrapper = Object.assign(new Error(auth.message, { cause: auth }), {
    isSshBootstrap: true,
    kind: auth.kind,
    sshError: auth.kind
  })

  assert.equal(isRetryableRemoteConnectionError(transientWrapper), true)
  assert.equal(isRetryableRemoteConnectionError(permanentWrapper), false)
})

test('offline and refused connections remain retryable ordinary transport errors', async () => {
  for (const error of [
    Object.assign(new Error('offline'), { code: 'ERR_INTERNET_DISCONNECTED' }),
    Object.assign(new Error('connection refused'), { code: 'ECONNREFUSED' })
  ]) {
    let attempts = 0

    const result = await resolveRemoteConnectionWithRetry(
      async () => {
        attempts += 1

        if (attempts === 1) {
          throw error
        }

        return 'connected'
      },
      { maxAttempts: 2, sleep: async () => undefined }
    )

    assert.equal(result, 'connected')
    assert.equal(attempts, 2)
  }
})

test('a permanently unresolved hostname is not retried', async () => {
  let attempts = 0
  const error = Object.assign(new Error('getaddrinfo ENOTFOUND invalid.example'), { code: 'ENOTFOUND' })

  await assert.rejects(
    () =>
      resolveRemoteConnectionWithRetry(
        async () => {
          attempts += 1
          throw error
        },
        { maxAttempts: 5, sleep: async () => undefined }
      ),
    rejected => rejected === error
  )

  assert.equal(attempts, 1)
})

test('ordinary connection retries are bounded', async () => {
  let attempts = 0
  const timeout = new Error('Timed out connecting to Hermes backend after 8000ms')

  await assert.rejects(
    () =>
      resolveRemoteConnectionWithRetry(
        async () => {
          attempts += 1
          throw timeout
        },
        { maxAttempts: 3, sleep: async () => undefined }
      ),
    error => error === timeout
  )

  assert.equal(attempts, 3)
})

test('permanent URL, token, and SSH configuration errors fail immediately', async () => {
  const permanentErrors = [
    Object.assign(new TypeError('Invalid URL'), { code: 'ERR_INVALID_URL' }),
    new Error('Remote Hermes gateway is selected, but no session token is saved.'),
    new Error('HERMES_DESKTOP_REMOTE_URL is set but HERMES_DESKTOP_REMOTE_TOKEN is not.'),
    Object.assign(new Error('Permission denied (publickey).'), { kind: 'auth-failed' }),
    Object.assign(new Error('Remote host identification has changed.'), { kind: 'host-key-changed' }),
    Object.assign(new Error('SSH bootstrap was superseded by newer connection settings.'), { kind: 'superseded' }),
    Object.assign(new Error('Unsupported SSH configuration.'), { kind: 'unknown' })
  ]

  for (const error of permanentErrors) {
    let attempts = 0

    assert.equal(isRetryableRemoteConnectionError(error), false)
    await assert.rejects(
      () =>
        resolveRemoteConnectionWithRetry(
          async () => {
            attempts += 1
            throw error
          },
          { maxAttempts: 5, sleep: async () => undefined }
        ),
      candidate => candidate === error
    )
    assert.equal(attempts, 1)
  }
})

test('wrapped transport failures remain retryable', () => {
  const wrapped = Object.assign(new Error('Could not reach the remote gateway.'), {
    cause: Object.assign(new Error('fetch failed'), { cause: Object.assign(new Error('refused'), { code: 'ECONNREFUSED' }) })
  })

  assert.equal(isRetryableRemoteConnectionError(wrapped), true)
})
