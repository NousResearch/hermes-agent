import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  ATTACHED_LIVENESS_FAILURE_THRESHOLD,
  classifyAttachedProbeError,
  createAttachedLivenessTracker,
  isTimeoutLikeError,
  isTransientWsProbeReason
} from './attached-backend-liveness'

test('a single transient probe failure does not exhaust the liveness budget', () => {
  const tracker = createAttachedLivenessTracker(3)

  assert.equal(tracker.noteTransientFailure(), false)
  assert.equal(tracker.consecutiveTransientFailures, 1)
})

test('consecutive transient failures declare gone only at the threshold', () => {
  const tracker = createAttachedLivenessTracker(3)

  assert.equal(tracker.noteTransientFailure(), false)
  assert.equal(tracker.noteTransientFailure(), false)
  assert.equal(tracker.noteTransientFailure(), true)
})

test('a success resets the transient streak so intermittent slowness never accumulates', () => {
  const tracker = createAttachedLivenessTracker(3)

  assert.equal(tracker.noteTransientFailure(), false)
  assert.equal(tracker.noteTransientFailure(), false)
  tracker.noteSuccess()
  assert.equal(tracker.consecutiveTransientFailures, 0)
  assert.equal(tracker.noteTransientFailure(), false)
  assert.equal(tracker.noteTransientFailure(), false)
  assert.equal(tracker.noteTransientFailure(), true)
})

test('a hard failure declares gone immediately', () => {
  const tracker = createAttachedLivenessTracker(3)

  assert.equal(tracker.noteHardFailure(), true)
  assert.equal(tracker.consecutiveTransientFailures, 0)
})

test('timeouts and 5xx probe errors classify as transient', () => {
  assert.equal(
    classifyAttachedProbeError(new Error('Timed out connecting to Hermes backend after 5000ms')),
    'transient'
  )
  assert.equal(classifyAttachedProbeError(new Error('socket hang up')), 'transient')
  assert.equal(classifyAttachedProbeError(new Error('503: Service Unavailable')), 'transient')
  assert.equal(classifyAttachedProbeError(new Error('fetch failed')), 'transient')
})

test('credentialed auth rejections classify as hard', () => {
  assert.equal(classifyAttachedProbeError(new Error('401: no_cookie')), 'hard')
  assert.equal(classifyAttachedProbeError(new Error('403: forbidden')), 'hard')
})

test('wrapped reauth errors classify as hard, not transient', () => {
  // The credentialed-probe shape: backend-health's makeReauthRequiredError
  // replaces the message, the 401 surviving only in .detail + reauth flags.
  const wrapped = Object.assign(
    new Error('Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.'),
    { needsOauthLogin: true, isReauthRequired: true, detail: '401: no_cookie' }
  )
  assert.equal(classifyAttachedProbeError(wrapped), 'hard')

  // Flags alone are enough even when the detail is missing.
  const flagsOnly = Object.assign(new Error('session expired'), { isReauthRequired: true })
  assert.equal(classifyAttachedProbeError(flagsOnly), 'hard')

  // A bare 401 in .detail with no flags is still an auth rejection.
  const detailOnly = Object.assign(new Error('probe failed'), { detail: '403: forbidden' })
  assert.equal(classifyAttachedProbeError(detailOnly), 'hard')

  // ...but an unflagged session-expiry-looking message stays transient.
  assert.equal(classifyAttachedProbeError(new Error('session expired, retrying')), 'transient')
})

test('timeout detection covers TimeoutError shapes and ETIMEDOUT codes', () => {
  assert.equal(isTimeoutLikeError(new Error('Timed out connecting to Hermes backend after 5000ms')), true)
  assert.equal(isTimeoutLikeError(Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' })), true)
  assert.equal(isTimeoutLikeError(new Error('401: unauthorized')), false)
})

test('transient WS probe reasons retry, auth rejections do not', () => {
  assert.equal(isTransientWsProbeReason('Timed out after 10000ms waiting for the WebSocket to open.'), true)
  assert.equal(isTransientWsProbeReason('WebSocket connection failed.'), true)
  assert.equal(isTransientWsProbeReason('unauthorized'), false)
  assert.equal(
    isTransientWsProbeReason('The gateway accepted the connection then closed it (credential rejected?)'),
    false
  )
})

test('default threshold matches the documented consecutive-failure policy', () => {
  assert.equal(ATTACHED_LIVENESS_FAILURE_THRESHOLD, 3)
  const tracker = createAttachedLivenessTracker()

  tracker.noteTransientFailure()
  tracker.noteTransientFailure()
  assert.equal(tracker.noteTransientFailure(), true)
})
