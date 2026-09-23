/**
 * Credential-free, provider-gated managed SSH acceptance cases.
 *
 * The current Electron build reports the trusted managed-rollout provider as
 * unavailable. These named skips are intentionally not release evidence. They
 * refuse arbitrary targets and never accept or store credentials.
 */

import type { TestInfo } from '@playwright/test'

import { selectManagedRolloutFixture } from './managed-rollout-fixtures'
import { test } from './test'

function requireCredentialFreeProvider(testInfo: TestInfo, browserName: string): void {
  const selected = selectManagedRolloutFixture(process.env)

  if (selected.state === 'refused') {
    testInfo.skip(true, selected.reason)

    return
  }

  testInfo.skip(true, `disposable selectors are configured, but no ${browserName} fixture provider is available`)
}

test.describe('managed SSH rollout acceptance (provider-gated)', () => {
  test('preparation completes before a pinned rollout can be admitted', async ({ browserName }, testInfo) => {
    requireCredentialFreeProvider(testInfo, browserName)
  })

  test('recheck observes an unknown attempt without redispatching it', async ({ browserName }, testInfo) => {
    requireCredentialFreeProvider(testInfo, browserName)
  })

  test('recovery requires the original correlation and positive clearance', async ({ browserName }, testInfo) => {
    requireCredentialFreeProvider(testInfo, browserName)
  })

  test('T21 keyboard/focus navigation keeps rollout actions reachable', async ({ browserName }, testInfo) => {
    requireCredentialFreeProvider(testInfo, browserName)
  })

  test('T21 RTL layout follows the selected localization direction', async ({ browserName }, testInfo) => {
    requireCredentialFreeProvider(testInfo, browserName)
  })

  test('T21 reduced-motion rendering settles without animation-dependent assertions', async ({ browserName }, testInfo) => {
    requireCredentialFreeProvider(testInfo, browserName)
  })

  test('T21 responsive scale rendering remains bounded at the managed-rollout viewport', async ({ browserName }, testInfo) => {
    requireCredentialFreeProvider(testInfo, browserName)
  })

  test('T21 unavailable-provider UI stays fail-closed without starting work', async ({ browserName }, testInfo) => {
    requireCredentialFreeProvider(testInfo, browserName)
  })
})
