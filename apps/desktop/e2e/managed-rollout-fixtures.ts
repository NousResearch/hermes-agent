/**
 * Select an explicitly disposable managed-rollout fixture without accepting
 * a hostname or credential value from the test runner's environment.
 *
 * These are opaque identifiers for a fixture provider. A configured selector
 * does not mean that a provider or an SSH target is available.
 */
import { execFile } from 'node:child_process'

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

/**
 * Disposable SSH fixture selection for the host journeys.
 *
 * The selector accepts only loopback endpoints, an absolute test-key path, and
 * a small shell-safe user. A configured selector means the caller may attempt a
 * real SSH journey; it never accepts a hostname, password, or credential value
 * from the runner's environment beyond the test key path.
 */
const FIXTURE_SET_ENV = 'HERMES_MANAGED_ROLLOUT_FIXTURE_SET'
const FIXTURE_KEY_ENV = 'HERMES_MANAGED_ROLLOUT_FIXTURE_KEY'
const FIXTURE_USER_ENV = 'HERMES_MANAGED_ROLLOUT_FIXTURE_USER'

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1'])
const FIXTURE_ENDPOINT = /^([A-Za-z0-9.:[\]-]+):([0-9]{1,5})$/
const FIXTURE_USER = /^[a-z_][a-z0-9_-]{0,31}$/
const MAX_FIXTURE_ENDPOINTS = 64

export interface ManagedRolloutSshEndpoint {
  host: string
  port: number
}

export type ManagedRolloutSshFixtureSelection =
  | { state: 'refused'; reason: string }
  | { state: 'configured'; endpoints: ManagedRolloutSshEndpoint[]; user: string; keyPath: string }

export function selectManagedRolloutSshFixtures(env: NodeJS.ProcessEnv): ManagedRolloutSshFixtureSelection {
  const raw = env[FIXTURE_SET_ENV]?.trim()

  if (!raw) {
    return { state: 'refused', reason: `${FIXTURE_SET_ENV} is unset; no disposable SSH fixture is selected` }
  }

  const items = raw.split(',').map(item => item.trim()).filter(Boolean)

  if (items.length === 0 || items.length > MAX_FIXTURE_ENDPOINTS) {
    return { state: 'refused', reason: `refusing malformed disposable SSH fixture set in ${FIXTURE_SET_ENV}` }
  }

  const endpoints: ManagedRolloutSshEndpoint[] = []

  for (const item of items) {
    const match = FIXTURE_ENDPOINT.exec(item)

    if (!match) {
      return { state: 'refused', reason: `refusing malformed disposable SSH endpoint in ${FIXTURE_SET_ENV}` }
    }

    if (!LOOPBACK_HOSTS.has(match[1])) {
      return { state: 'refused', reason: `refusing non-disposable SSH fixture host in ${FIXTURE_SET_ENV}` }
    }

    const port = Number(match[2])

    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      return { state: 'refused', reason: `refusing invalid disposable SSH fixture port in ${FIXTURE_SET_ENV}` }
    }

    endpoints.push({ host: match[1], port })
  }

  if (new Set(endpoints.map(endpoint => `${endpoint.host}:${endpoint.port}`)).size !== endpoints.length) {
    return { state: 'refused', reason: `refusing duplicate disposable SSH fixture endpoints in ${FIXTURE_SET_ENV}` }
  }

  const keyPath = env[FIXTURE_KEY_ENV]?.trim()

  if (!keyPath) {
    return { state: 'refused', reason: `${FIXTURE_KEY_ENV} is unset; no disposable fixture key is selected` }
  }

  if (!/^(?:[A-Za-z]:[\\/]|\/)/.test(keyPath)) {
    return { state: 'refused', reason: `refusing non-absolute fixture key path in ${FIXTURE_KEY_ENV}` }
  }

  const user = env[FIXTURE_USER_ENV]?.trim() || 'fixture'

  if (!FIXTURE_USER.test(user)) {
    return { state: 'refused', reason: `refusing invalid disposable fixture user in ${FIXTURE_USER_ENV}` }
  }

  return { state: 'configured', endpoints, user, keyPath }
}

/**
 * Real SSH transport for a selected disposable fixture. Every command runs
 * with BatchMode so a key rejection fails fast instead of prompting, and the
 * endpoint was already constrained to loopback by the selector.
 */
export function createFixtureSshExec(
  endpoint: ManagedRolloutSshEndpoint & { user: string; keyPath: string }
): (command: string, options?: { timeoutMs?: number }) => Promise<string> {
  return (command, options = {}) =>
    new Promise<string>((resolve, reject) => {
      execFile(
        'ssh',
        [
          '-i', endpoint.keyPath,
          '-p', String(endpoint.port),
          '-o', 'BatchMode=yes',
          '-o', 'StrictHostKeyChecking=no',
          '-o', 'LogLevel=ERROR',
          `${endpoint.user}@${endpoint.host}`,
          '--',
          command
        ],
        { timeout: options.timeoutMs ?? 30_000, windowsHide: true, maxBuffer: 8 * 1024 * 1024 },
        (error, stdout, stderr) => {
          if (error) {
            const detail = String(stderr || '').trim().slice(0, 500)

            reject(new Error(`disposable fixture ssh failed: ${error.message}${detail ? ` :: ${detail}` : ''}`))

            return
          }

          resolve(String(stdout))
        }
      )
    })
}
