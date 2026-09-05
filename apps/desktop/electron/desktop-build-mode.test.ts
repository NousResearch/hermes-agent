import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  assertConnectionModeAllowed,
  createRemoteConnectionGate,
  formatBackendErrorForLog,
  isConnectionKindAllowedForRemoteOnly,
  sanitizeRemoteSetupError,
  sanitizeRemoteSetupFailure,
  shouldResumeRemoteConnectionGate
} from './desktop-build-mode'

test('remote-only builds reject local and SSH connection modes', () => {
  assert.throws(() => assertConnectionModeAllowed('local', true), /requires a remote Hermes connection/)
  assert.throws(() => assertConnectionModeAllowed('ssh', true), /requires a remote Hermes connection/)
  assert.doesNotThrow(() => assertConnectionModeAllowed('remote', true))
  assert.doesNotThrow(() => assertConnectionModeAllowed('cloud', true))
  assert.doesNotThrow(() => assertConnectionModeAllowed('local', false))
  assert.equal(isConnectionKindAllowedForRemoteOnly('local', true), false)
  assert.equal(isConnectionKindAllowedForRemoteOnly('ssh', true), false)
  assert.equal(isConnectionKindAllowedForRemoteOnly('remote', true), true)
  assert.equal(isConnectionKindAllowedForRemoteOnly('cloud', true), true)
  assert.equal(isConnectionKindAllowedForRemoteOnly('local', false), true)
})

test('remote connection gate shares one waiter and resumes it once', async () => {
  const gate = createRemoteConnectionGate()
  const first = gate.wait()
  const second = gate.wait()

  assert.equal(first, second)
  assert.equal(gate.hasWaiter(), true)

  gate.resume()
  await first

  assert.equal(gate.hasWaiter(), false)
})

test('only a global remote apply resumes first-run setup', () => {
  assert.equal(shouldResumeRemoteConnectionGate(true, null, true), true)
  assert.equal(shouldResumeRemoteConnectionGate(true, 'work', true), false)
  assert.equal(shouldResumeRemoteConnectionGate(true, null, false), false)
  assert.equal(shouldResumeRemoteConnectionGate(false, null, true), false)
})

test('remote setup errors expose only coarse categories', () => {
  const tokenError = sanitizeRemoteSetupError(new Error('request https://gateway.example/?token=secret failed (401)'))

  assert.equal(tokenError, 'Remote gateway authentication needs attention.')
  assert.equal(tokenError.includes('secret'), false)
  assert.equal(
    sanitizeRemoteSetupError(new Error('malformed URL https://gateway.example/?ticket=secret')),
    'Remote gateway URL needs attention.'
  )
  assert.equal(sanitizeRemoteSetupError(new Error('ECONNRESET')), 'Could not connect to the remote Hermes gateway.')
})

test('remote-only backend logs never include raw transport details', () => {
  const raw = new Error('GET https://gateway.example/?token=secret failed: response body password=secret')
  const remoteLog = formatBackendErrorForLog(raw, true)

  assert.equal(remoteLog, 'Remote gateway authentication needs attention.')
  assert.equal(remoteLog.includes('gateway.example'), false)
  assert.equal(remoteLog.includes('secret'), false)
  assert.equal(formatBackendErrorForLog(raw, false), raw.message)
  assert.match(formatBackendErrorForLog(raw, false, true), /gateway\.example/)
})

test('remote-only backend failures retain safe classification without retaining the cause', () => {
  const raw = Object.assign(new Error('POST https://gateway.example/?token=secret returned 401 body=secret'), {
    isCloudBackendDown: true,
    needsOauthLogin: true,
    statusCode: 401
  })

  const safe = sanitizeRemoteSetupFailure(raw)

  assert.equal(safe.message, 'Remote gateway authentication needs attention.')
  assert.equal(safe.isCloudBackendDown, true)
  assert.equal(safe.needsOauthLogin, true)
  assert.equal(safe.statusCode, 401)
  assert.equal('cause' in safe, false)
  assert.equal(safe.stack?.includes('gateway.example'), false)
  assert.equal(safe.stack?.includes('secret'), false)
})
