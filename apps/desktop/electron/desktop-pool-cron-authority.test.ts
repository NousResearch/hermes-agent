import assert from 'node:assert/strict'
import { poolBackendAuthorityEnv } from './desktop-pool-cron-authority'

import { test } from 'vitest'

test('pooled Desktop profile backends are marked as non-authoritative cron workers', () => {
  assert.deepEqual(poolBackendAuthorityEnv, {
    HERMES_DESKTOP: '1',
    HERMES_DESKTOP_POOL: '1'
  })
})
