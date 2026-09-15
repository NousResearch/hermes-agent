import assert from 'node:assert/strict'

import { test } from 'vitest'

import { ensurePoolBackendRuntime } from './pool-backend-startup'

test('a background bootstrap-needed resolution cannot start recovery or installation', async () => {
  const backend = {
    kind: 'bootstrap-needed',
    label: 'Hermes Agent not installed yet; bootstrap required',
    command: null,
    args: ['--profile', 'worker', 'serve', '--host', '127.0.0.1', '--port', '0'],
    bootstrap: true,
    env: {},
    shell: false,
    activeRoot: '/test/hermes-agent',
    installStamp: null,
    isPackaged: false,
    platform: process.platform
  }

  const effects: string[] = []

  const result = await ensurePoolBackendRuntime({
    backend,
    profile: 'worker',
    ensureRuntime: async candidate => {
      // This boundary owns both installer paths in main.ts. Entering it on
      // the unresolved sentinel is already a background recovery side effect.
      effects.push('recovery-handoff', 'bootstrap-start')

      return { ...candidate, command: 'hermes' }
    }
  }).then(
    value => ({ value, error: null }),
    (error: unknown) => ({ value: null, error })
  )

  assert.deepEqual(effects, [], 'background resolution must not enter installer preparation')
  assert.equal(result.value, null)
  assert.ok(result.error instanceof Error)
  assert.match(result.error.message, /worker/)
  assert.match(result.error.message, /install|repair/i)
})

for (const bootstrap of [false, true]) {
  test(`a ready background runtime still runs preparation when bootstrap=${bootstrap}`, async () => {
    const backend = {
      kind: 'python',
      label: 'Hermes runtime',
      command: '/test/python',
      args: ['--profile', 'worker', 'serve', '--port', '0'],
      bootstrap,
      env: {},
      shell: false
    }

    const preparedBackend = { ...backend, command: '/test/venv/python' }
    const prepared: (typeof backend)[] = []

    const result = await ensurePoolBackendRuntime({
      backend,
      profile: 'worker',
      ensureRuntime: async candidate => {
        prepared.push(candidate)

        return preparedBackend
      }
    })

    assert.equal(result, preparedBackend)
    assert.deepEqual(prepared, [backend])
  })
}

test('background runtime preparation failures propagate without a retry', async () => {
  const failure = new Error('Hermes venv is missing; repair is required.')
  let attempts = 0

  await assert.rejects(
    ensurePoolBackendRuntime({
      backend: { kind: 'python', bootstrap: true },
      profile: 'worker',
      ensureRuntime: async () => {
        attempts += 1
        throw failure
      }
    }),
    error => error === failure
  )
  assert.equal(attempts, 1)
})
