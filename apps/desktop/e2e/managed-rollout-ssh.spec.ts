/**
 * Credential-free, provider-gated managed SSH acceptance cases.
 *
 * The current Electron build reports the trusted managed-rollout provider as
 * unavailable. These named skips are intentionally not release evidence. They
 * refuse arbitrary targets and never accept or store credentials.
 */

import type { TestInfo } from '@playwright/test'

import { test } from './test'

const TEST_TARGET_ENV = 'HERMES_MANAGED_ROLLOUT_E2E_TARGET'
const SAFE_TEST_TARGET = /^test:\/\/hermes-managed-rollout(?:\/[A-Za-z0-9._-]+)?$/

function requireCredentialFreeProvider(testInfo: TestInfo): void {
  const target = process.env[TEST_TARGET_ENV]

  if (!target) {
    testInfo.skip(true, `${TEST_TARGET_ENV} is unset; no credential-free managed-rollout provider fixture is configured`)

    return
  }

  if (!SAFE_TEST_TARGET.test(target)) {
    testInfo.skip(true, `refusing non-test managed-rollout target; use the explicit ${TEST_TARGET_ENV} test:// scheme`)

    return
  }

  testInfo.skip(true, 'managed-rollout provider is unavailable in the current Electron main process')
}

test.describe('managed SSH rollout acceptance (provider-gated)', () => {
  test('preparation completes before a pinned rollout can be admitted', async (_fixtures, testInfo) => {
    requireCredentialFreeProvider(testInfo)
  })

  test('recheck observes an unknown attempt without redispatching it', async (_fixtures, testInfo) => {
    requireCredentialFreeProvider(testInfo)
  })

  test('recovery requires the original correlation and positive clearance', async (_fixtures, testInfo) => {
    requireCredentialFreeProvider(testInfo)
  })

  test('T21 keyboard/focus navigation keeps rollout actions reachable', async (_fixtures, testInfo) => {
    requireCredentialFreeProvider(testInfo)
  })

  test('T21 RTL layout follows the selected localization direction', async (_fixtures, testInfo) => {
    requireCredentialFreeProvider(testInfo)
  })

  test('T21 reduced-motion rendering settles without animation-dependent assertions', async (_fixtures, testInfo) => {
    requireCredentialFreeProvider(testInfo)
  })

  test('T21 responsive scale rendering remains bounded at the managed-rollout viewport', async (_fixtures, testInfo) => {
    requireCredentialFreeProvider(testInfo)
  })

  test('T21 unavailable-provider UI stays fail-closed without starting work', async (_fixtures, testInfo) => {
    requireCredentialFreeProvider(testInfo)
  })
})
