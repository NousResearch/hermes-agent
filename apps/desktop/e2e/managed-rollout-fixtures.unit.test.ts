import { describe, expect, it } from 'vitest'

import {
  createFixtureSshExec,
  selectManagedRolloutFixture,
  selectManagedRolloutSshFixtures
} from './managed-rollout-fixtures'

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

describe('disposable SSH fixture selection matrix', () => {
  const key = 'C:/Users/example/.hermes-fixture/fixture_key'

  it('refuses an unset fixture set and an unset key before any connection', () => {
    expect(selectManagedRolloutSshFixtures({})).toEqual({
      state: 'refused',
      reason: 'HERMES_MANAGED_ROLLOUT_FIXTURE_SET is unset; no disposable SSH fixture is selected'
    })

    expect(selectManagedRolloutSshFixtures({ HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1:2222' })).toEqual({
      state: 'refused',
      reason: 'HERMES_MANAGED_ROLLOUT_FIXTURE_KEY is unset; no disposable fixture key is selected'
    })
  })

  it('refuses a non-loopback host so no production endpoint can be selected', () => {
    expect(
      selectManagedRolloutSshFixtures({
        HERMES_MANAGED_ROLLOUT_FIXTURE_SET: 'ssh.example.com:22',
        HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key
      })
    ).toEqual({ state: 'refused', reason: 'refusing non-disposable SSH fixture host in HERMES_MANAGED_ROLLOUT_FIXTURE_SET' })
  })

  it('refuses malformed endpoints, invalid ports, and duplicate endpoints', () => {
    expect(
      selectManagedRolloutSshFixtures({
        HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1',
        HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key
      }).state
    ).toBe('refused')

    expect(
      selectManagedRolloutSshFixtures({
        HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1:0',
        HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key
      }).state
    ).toBe('refused')

    expect(
      selectManagedRolloutSshFixtures({
        HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1:70000',
        HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key
      }).state
    ).toBe('refused')

    expect(
      selectManagedRolloutSshFixtures({
        HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1:2222,127.0.0.1:2222',
        HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key
      })
    ).toEqual({ state: 'refused', reason: 'refusing duplicate disposable SSH fixture endpoints in HERMES_MANAGED_ROLLOUT_FIXTURE_SET' })
  })

  it('refuses a relative key path and an unsafe user', () => {
    expect(
      selectManagedRolloutSshFixtures({
        HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1:2222',
        HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: 'fixture_key'
      })
    ).toEqual({ state: 'refused', reason: 'refusing non-absolute fixture key path in HERMES_MANAGED_ROLLOUT_FIXTURE_KEY' })

    expect(
      selectManagedRolloutSshFixtures({
        HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1:2222',
        HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key,
        HERMES_MANAGED_ROLLOUT_FIXTURE_USER: 'root; rm -rf /'
      }).state
    ).toBe('refused')
  })

  it('accepts a bounded loopback set and defaults the fixture user', () => {
    expect(
      selectManagedRolloutSshFixtures({
        HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1:2222,127.0.0.1:2223',
        HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key
      })
    ).toEqual({
      state: 'configured',
      endpoints: [
        { host: '127.0.0.1', port: 2222 },
        { host: '127.0.0.1', port: 2223 }
      ],
      user: 'fixture',
      keyPath: key
    })
  })

  it('refuses a fixture set above the endpoint bound', () => {
    const oversized = Array.from({ length: 65 }, (_, index) => `127.0.0.1:${2200 + index}`).join(',')

    expect(
      selectManagedRolloutSshFixtures({
        HERMES_MANAGED_ROLLOUT_FIXTURE_SET: oversized,
        HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key
      })
    ).toEqual({ state: 'refused', reason: 'refusing malformed disposable SSH fixture set in HERMES_MANAGED_ROLLOUT_FIXTURE_SET' })
  })

  it('fails closed instead of returning output when a declared fixture cannot be reached', async () => {
    const exec = createFixtureSshExec({ host: '127.0.0.1', port: 1, user: 'fixture', keyPath: key })

    await expect(exec('echo unreachable')).rejects.toThrow(/disposable fixture ssh failed/)
  })
})
