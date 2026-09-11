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
 * Composes the REAL modules exactly the way main.ts does, so the contract is
 * proven on the code that ships rather than on a mock of it.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { httpStatusError } from './api-transport'
import { isReauthRequiredError } from './backend-health'
import { isRetryableRemoteBootFailure, shouldLatchRemoteReauthFailure } from './backend-start-failure'
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
