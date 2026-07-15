import assert from 'node:assert/strict'

import { test } from 'vitest'

import { requiresOauthLogin, resolveRemoteConnectionWithRetry } from './remote-connection-retry'

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
