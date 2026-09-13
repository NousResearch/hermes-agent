import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  poolBackendAuthorityEnv,
  withPoolBackendAuthorityEnv
} from './desktop-pool-cron-authority'

test('pooled Desktop profile backends are marked as non-authoritative cron workers', () => {
  const env = withPoolBackendAuthorityEnv({ HERMES_DESKTOP: '0', CUSTOM: 'kept' })

  assert.equal(env.HERMES_DESKTOP, poolBackendAuthorityEnv.HERMES_DESKTOP)
  assert.equal(env.HERMES_DESKTOP_POOL, poolBackendAuthorityEnv.HERMES_DESKTOP_POOL)
  assert.equal(env.CUSTOM, 'kept')
})
