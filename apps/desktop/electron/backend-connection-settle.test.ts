import assert from 'node:assert/strict'

import { test } from 'vitest'

import { replacementBackendPromise, SUPERSEDED_BACKEND_START } from './backend-connection-settle'

test('a superseded attempt adopts the replacement connection', async () => {
  let current: Promise<string> | null = null
  const own = Promise.resolve('old')
  const replacement = Promise.resolve('new')
  let polls = 0

  const adopted = await replacementBackendPromise(
    own,
    () => {
      polls += 1

      return polls < 2 ? current : replacement
    },
    500,
    async () => {
      current = replacement
    }
  )

  assert.equal(await adopted, 'new')
  assert.equal(SUPERSEDED_BACKEND_START.includes('superseded'), true)
})

test('no replacement leaves the stale attempt unresolved', async () => {
  const own = Promise.resolve('old')

  const adopted = await replacementBackendPromise(own, () => own, 0, async () => undefined)

  assert.equal(adopted, null)
})
