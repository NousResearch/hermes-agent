import { describe, expect, it } from 'vitest'

import { selectManagedRolloutFixture } from './managed-rollout-fixtures'

describe('disposable managed-rollout fixture selection', () => {
  const target = 'test://hermes-managed-rollout/two-hosts'
  const credentialRef = 'test-credential://hermes-managed-rollout/lease-1'

  it('refuses an arbitrary SSH target before reading a credential selector', () => {
    expect(selectManagedRolloutFixture({ HERMES_MANAGED_ROLLOUT_E2E_TARGET: 'ssh://production.example' })).toEqual({
      state: 'refused',
      reason: 'refusing non-test managed-rollout target in HERMES_MANAGED_ROLLOUT_E2E_TARGET'
    })
  })

  it('refuses a disposable target without a test credential reference', () => {
    expect(selectManagedRolloutFixture({ HERMES_MANAGED_ROLLOUT_E2E_TARGET: target })).toEqual({
      state: 'refused',
      reason: 'HERMES_MANAGED_ROLLOUT_E2E_CREDENTIAL_REF is unset; no test credential reference is selected'
    })
  })

  it('refuses raw credential values and accepts only an opaque test reference', () => {
    expect(
      selectManagedRolloutFixture({
        HERMES_MANAGED_ROLLOUT_E2E_TARGET: target,
        HERMES_MANAGED_ROLLOUT_E2E_CREDENTIAL_REF: '-----BEGIN PRIVATE KEY-----'
      }).state
    ).toBe('refused')

    expect(
      selectManagedRolloutFixture({
        HERMES_MANAGED_ROLLOUT_E2E_TARGET: target,
        HERMES_MANAGED_ROLLOUT_E2E_CREDENTIAL_REF: credentialRef
      })
    ).toEqual({ state: 'configured', target, credentialRef })
  })
})
