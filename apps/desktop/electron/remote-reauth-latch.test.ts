/**
 * Regression suite for issue #95701: an expired remote OAuth session must
 * boot into ONE latched recovery overlay, not flicker between the connecting
 * state and the overlay while the renderer re-drives a rejection that can
 * never self-heal.
 *
 * The chain that broke:
 *
 *   fetchJson (native bearer)  →  bare Error("401: ...") — no statusCode
 *   withTransientRetries       →  not an auth rejection → hammered 3x
 *   gatewayTicketFailure       →  transport copy, no needsOauthLogin
 *   startHermes                →  isReauth=false → NOT latched, retryable:true
 *   renderer boot-retry loop   →  running:true hides the overlay, repeat
 *
 * The first half of this file composes the REAL modules exactly the way
 * main.ts does, so the contract is proven on the code that ships rather than
 * on a mock of it. The second half pins the main.ts wiring with the repo's
 * source-assertion pattern (main.ts has no exports; see hardening.test.ts).
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { httpStatusError } from './api-transport'
import { isReauthRequiredError } from './backend-health'
import {
  isRetryableRemoteBootFailure,
  shouldHoldBootProgressForReauth,
  shouldLatchRemoteReauthFailure
} from './backend-start-failure'
import { gatewayTicketFailure, isGatewayAuthRejection, withTransientRetries } from './connection-config'
import { shouldRotateNativeTokenAfterRejection } from './native-auth-decisions'

// --- composition: the real modules, in production order ------------------

test('FIX #95701: a native-bearer 401 is a confirmed, non-retryable reauth rejection end to end', async () => {
  // What fetchJson now throws for the gate's structured session_expired 401.
  const bearerRejection = httpStatusError(401, '{"error":"session_expired","reason":"invalid_or_expired_session"}')

  assert.equal(isGatewayAuthRejection(bearerRejection), true)

  // mintGatewayWsTicket's transient-retry wrapper fails immediately: a dead
  // session is never hammered.
  let attempts = 0

  const mintError = await withTransientRetries(
    async () => {
      attempts += 1
      throw bearerRejection
    },
    { sleep: async () => {} }
  ).then(
    () => null,
    (error: unknown) => error
  )

  assert.equal(attempts, 1)
  assert.equal(mintError, bearerRejection)

  // buildRemoteConnection wraps the rejection for the boot path.
  const wrapped = gatewayTicketFailure(mintError, 'session expired — sign in', 'could not reach gateway') as any

  assert.equal(wrapped.message, 'session expired — sign in')
  assert.equal(wrapped.needsOauthLogin, true)
  assert.equal(wrapped.statusCode, 401)

  // startHermes's own composition: isReauth = isReauthRequiredError(error).
  const isReauth = isReauthRequiredError(wrapped)

  assert.equal(isReauth, true)
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth }), true)
  assert.equal(
    isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth }),
    false,
    'the boot progress must be non-retryable so the renderer never re-drives the boot'
  )
})

test('FIX #95701: the pre-fix error shape is exactly what made the rejection look transient', async () => {
  // A "401: ..." message with no structured statusCode — what fetchJson used
  // to throw. Every classifier below is shape-based on purpose, so this is
  // the regression the structured error prevents.
  const anonymous = new Error('401: {"error":"session_expired"}')

  assert.equal(isGatewayAuthRejection(anonymous), false)
  assert.equal(shouldRotateNativeTokenAfterRejection(anonymous), false)

  let attempts = 0

  await withTransientRetries(
    async () => {
      attempts += 1
      throw anonymous
    },
    { attempts: 3, sleep: async () => {} }
  ).catch(() => undefined)

  assert.equal(attempts, 3, 'an anonymous 401 was retried like a transport blip')

  const wrapped = gatewayTicketFailure(anonymous, 'auth copy', 'transport copy')

  assert.equal(wrapped.message, 'transport copy')
  assert.equal(isReauthRequiredError(wrapped), false)
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: isReauthRequiredError(wrapped) }), true)
})

test('FIX #95701: a bearer 401 earns exactly one forced rotation; a 403 or a 5xx does not', () => {
  assert.equal(shouldRotateNativeTokenAfterRejection(httpStatusError(401, 'expired')), true)
  assert.equal(shouldRotateNativeTokenAfterRejection(httpStatusError(403, 'forbidden')), false)
  assert.equal(shouldRotateNativeTokenAfterRejection(httpStatusError(503, 'down')), false)

  // A 403 is still a confirmed rejection for the boot path — it just skips
  // the pointless rotation on the way there.
  const forbidden = gatewayTicketFailure(httpStatusError(403, 'forbidden'), 'auth copy', 'transport copy')

  assert.equal(isReauthRequiredError(forbidden), true)
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: true }), false)
})

test('FIX #95701: once latched, the boot surface ignores everything but the latched failure', () => {
  const latched = gatewayTicketFailure(httpStatusError(401, 'expired'), 'session expired — sign in', 'transport')

  // The latching emit itself (startHermes: updateBootProgress({ error: message, retryable: false })).
  assert.equal(shouldHoldBootProgressForReauth(latched.message, { error: latched.message }), false)
  // A sibling attempt's progress phase (advanceBootProgress → error: null, running: true).
  assert.equal(shouldHoldBootProgressForReauth(latched.message, { error: null }), true)
  // A sibling attempt's unrelated transport failure (would flip retryable back on).
  assert.equal(shouldHoldBootProgressForReauth(latched.message, { error: 'Desktop boot failed: transport' }), true)
})

// --- main.ts wiring ------------------------------------------------------

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

/** Source of one top-level function of main.ts, from its header to the next top-level function. */
function mainFunction(header: string): string {
  const start = mainSource.indexOf(header)

  assert.notEqual(start, -1, `${header} must exist in main.ts`)

  const candidates = ['\nfunction ', '\nasync function ']
    .map(marker => mainSource.indexOf(marker, start + header.length))
    .filter(index => index !== -1)

  const end = candidates.length > 0 ? Math.min(...candidates) : mainSource.length

  return mainSource.slice(start, end)
}

test('main.ts: fetchJson and fetchPublicJson throw the structured httpStatusError, never a bare Error', () => {
  for (const header of ['function fetchJson(', 'function fetchPublicJson(']) {
    const body = mainFunction(header)

    assert.match(body, /reject\(httpStatusError\(res\.statusCode, text, res\.statusMessage\)\)/, header)
    assert.doesNotMatch(
      body,
      /new Error\(`\$\{res\.statusCode\}:/,
      `${header} must not rebuild the anonymous status error`
    )
  }
})

test('main.ts: the native ticket mint forces one refresh before the rejection is confirmed', () => {
  const mint = mainFunction('async function mintGatewayWsTicket(')

  assert.match(mint, /if \(!shouldRotateNativeTokenAfterRejection\(error\)\) \{\s*throw error/)
  assert.match(mint, /ensureNativeAccessToken\(baseUrl, \{ forceRefresh: true \}\)/)
  assert.match(mint, /return await mintGatewayWsTicketWithBearer\(baseUrl, rotatedAt, headers\)/)

  const ensure = mainFunction('async function ensureNativeAccessToken(')

  assert.match(ensure, /if \(!options\.forceRefresh && !tokenNeedsRefresh\(/)
  // A dead refresh token drops the stored set (the overlay then reads "not
  // connected" and offers Sign in) — this branch only works because the
  // refresh POST now carries a structured statusCode.
  assert.match(ensure, /error\.statusCode === 401\) \{\s*_clearNativeTokens\(baseUrl\)/)
})

test('main.ts: startHermes latches the failure before its first yield back to the event loop', () => {
  const setPromise = mainSource.indexOf('backendConnectionState.setPromise(connectionAttempt, connectionPromise)')

  assert.notEqual(setPromise, -1)

  const catchStart = mainSource.lastIndexOf('.catch(async error => {', setPromise)

  assert.notEqual(catchStart, -1)

  const failurePath = mainSource.slice(catchStart, setPromise)
  const invalidate = failurePath.indexOf('backendConnectionState.invalidate()')
  const resetGuard = failurePath.indexOf('error instanceof FirstRunSetupResetError')
  const reauthLatch = failurePath.indexOf('remoteReauthFailure = error instanceof Error ? error : new Error(message)')
  const localLatch = failurePath.indexOf('backendStartFailure = error instanceof Error ? error : new Error(message)')
  const exitWait = failurePath.lastIndexOf('await waitForBackendExit(failedProcess)')
  const emit = failurePath.indexOf('updateBootProgress(')

  for (const [label, index] of Object.entries({ invalidate, resetGuard, reauthLatch, localLatch, exitWait, emit })) {
    assert.notEqual(index, -1, `${label} must exist in the startHermes failure path`)
  }

  // No await between invalidate() (which drops the shared attempt promise)
  // and the latch assignments — a concurrent caller must hit the latch, not
  // start a fresh attempt that re-emits running:true over the overlay.
  const beforeLatch = failurePath.slice(invalidate, Math.max(reauthLatch, localLatch))
  const awaitsBeforeLatch = beforeLatch.match(/\bawait\b/g) ?? []

  assert.equal(
    awaitsBeforeLatch.length,
    1,
    'the only await before the latches is the FirstRunSetupResetError early-exit branch'
  )
  assert.ok(resetGuard < reauthLatch, 'a first-run reset is never latched as a failure')
  assert.ok(reauthLatch < exitWait && localLatch < exitWait, 'latches are set before the exit wait yields')
  assert.ok(exitWait < emit, 'the failure is emitted after the child has exited, as before')
})

test('main.ts: updateBootProgress holds every non-latched update while the reauth latch is set', () => {
  const body = mainFunction('function updateBootProgress(')

  const hold = body.indexOf(
    'shouldHoldBootProgressForReauth(remoteReauthFailure ? remoteReauthFailure.message : null, update)'
  )

  const assign = body.indexOf('bootProgressState = {')

  assert.notEqual(hold, -1, 'the hold must key on the latched failure message')
  assert.notEqual(assign, -1)
  assert.ok(hold < assign, 'the hold runs before the state is touched or broadcast')
  assert.match(body.slice(hold, assign), /return\s*\}/, 'a held update returns without broadcasting')
})
