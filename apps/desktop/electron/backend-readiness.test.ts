import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  completeLocalBackendHandshake,
  createHermesReadinessProbe,
  shouldFallbackFromReadyEndpoint
} from './backend-readiness'

test('local backend readiness uses the lightweight endpoint', async () => {
  const paths: string[] = []

  const probe = createHermesReadinessProbe({
    preferReadyEndpoint: true,
    request: async path => {
      paths.push(path)
    }
  })

  await probe()

  assert.deepEqual(paths, ['/api/ready'])
})

test('remote readiness keeps the public status contract', async () => {
  const paths: string[] = []

  const probe = createHermesReadinessProbe({
    preferReadyEndpoint: false,
    request: async path => {
      paths.push(path)
    }
  })

  await probe()

  assert.deepEqual(paths, ['/api/status'])
})

test('an older backend falls back from a missing ready endpoint and remembers the fallback', async () => {
  const paths: string[] = []

  const probe = createHermesReadinessProbe({
    preferReadyEndpoint: true,
    request: async path => {
      paths.push(path)

      if (path === '/api/ready') {
        throw new Error('404: {"detail":"Not Found"}')
      }
    }
  })

  await probe()
  await probe()

  assert.deepEqual(paths, ['/api/ready', '/api/status', '/api/status'])
})

test('an old local runtime with token drift falls back to public status exactly once', async () => {
  const paths: string[] = []

  const probe = createHermesReadinessProbe({
    preferReadyEndpoint: true,
    request: async path => {
      paths.push(path)

      if (path === '/api/ready') {
        throw new Error('401: {"detail":"Unauthorized"}')
      }
    }
  })

  await probe()
  await probe()

  assert.deepEqual(paths, ['/api/ready', '/api/status', '/api/status'])
})

test('an older SPA fallback response is treated as a missing ready endpoint', () => {
  assert.equal(
    shouldFallbackFromReadyEndpoint(
      new Error(
        'Expected JSON from http://127.0.0.1:9119/api/ready but got HTML (status 200). ' +
          'The endpoint is likely missing on the Hermes backend.'
      )
    ),
    true
  )
})

test('local handshake adopts the served token only after readiness succeeds', async () => {
  const sequence: string[] = []

  const token = await completeLocalBackendHandshake({
    waitForReady: async () => {
      sequence.push('ready-via-public-status-fallback')
    },
    markReady: () => {
      sequence.push('mark-ready')
    },
    adoptServedToken: async () => {
      sequence.push('adopt-served-token')

      return 'served-token'
    }
  })

  assert.equal(token, 'served-token')
  assert.deepEqual(sequence, ['ready-via-public-status-fallback', 'mark-ready', 'adopt-served-token'])
})

test('transient ready failures retry ready instead of silently switching contracts', async () => {
  const paths: string[] = []

  const probe = createHermesReadinessProbe({
    preferReadyEndpoint: true,
    request: async path => {
      paths.push(path)
      throw new Error('503: starting')
    }
  })

  await assert.rejects(probe(), /503: starting/)
  await assert.rejects(probe(), /503: starting/)

  assert.deepEqual(paths, ['/api/ready', '/api/ready'])
})
