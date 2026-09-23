/**
 * Select an explicitly disposable managed-rollout fixture without accepting
 * a hostname or credential value from the test runner's environment.
 *
 * These are opaque identifiers for a fixture provider. A configured selector
 * does not mean that a provider or an SSH target is available.
 */
const TARGET_ENV = 'HERMES_MANAGED_ROLLOUT_E2E_TARGET'
const CREDENTIAL_REF_ENV = 'HERMES_MANAGED_ROLLOUT_E2E_CREDENTIAL_REF'

const TEST_TARGET = /^test:\/\/hermes-managed-rollout\/[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const TEST_CREDENTIAL_REF = /^test-credential:\/\/hermes-managed-rollout\/[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/

export type ManagedRolloutFixtureSelection =
  | { state: 'refused'; reason: string }
  | { state: 'configured'; target: string; credentialRef: string }

export function selectManagedRolloutFixture(env: NodeJS.ProcessEnv): ManagedRolloutFixtureSelection {
  const target = env[TARGET_ENV]?.trim()

  if (!target) {
    return { state: 'refused', reason: `${TARGET_ENV} is unset; no disposable target is selected` }
  }

  if (!TEST_TARGET.test(target)) {
    return { state: 'refused', reason: `refusing non-test managed-rollout target in ${TARGET_ENV}` }
  }

  const credentialRef = env[CREDENTIAL_REF_ENV]?.trim()

  if (!credentialRef) {
    return { state: 'refused', reason: `${CREDENTIAL_REF_ENV} is unset; no test credential reference is selected` }
  }

  if (!TEST_CREDENTIAL_REF.test(credentialRef)) {
    return { state: 'refused', reason: `refusing non-test credential reference in ${CREDENTIAL_REF_ENV}` }
  }

  return { state: 'configured', target, credentialRef }
}
