import { beforeEach, expect, test } from 'vitest'

import {
  $inAppToastsEnabled,
  $notifications,
  clearNotifications,
  gatewayErrorToastId,
  isDiskFullErrorMessage,
  notify,
  notifyError,
  setInAppToastsEnabled
} from './notifications'

beforeEach(() => {
  clearNotifications()
  setInAppToastsEnabled(true)
})

function lastMessage(): string {
  return $notifications.get()[0]?.message ?? ''
}

// Regression for #39365: a gateway auth 401 (bad API_SERVER_KEY) must not be
// summarized as a provider (OpenAI/OpenRouter) API key problem.
test('gateway_auth_failed error is summarized as gateway auth, not provider key', () => {
  notifyError(
    new Error(
      '401 {"error": {"message": "Invalid gateway API key (API_SERVER_KEY)", "type": "gateway_auth_error", "code": "gateway_auth_failed"}}'
    ),
    'Request failed'
  )

  expect(lastMessage()).toContain('API_SERVER_KEY')
  expect(lastMessage()).not.toMatch(/OpenAI/i)
})

test('provider invalid_api_key error still maps to the OpenAI summary', () => {
  notifyError(
    new Error('401 {"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}'),
    'Request failed'
  )

  expect(lastMessage()).toMatch(/OpenAI rejected the API key/i)
})

test('disk-full / ENOSPC errors toast a free-space message', () => {
  expect(isDiskFullErrorMessage('OSError: [Errno 28] No space left on device')).toBe(true)
  expect(isDiskFullErrorMessage('sqlite3.OperationalError: database or disk is full')).toBe(true)
  expect(isDiskFullErrorMessage('disk full: session storage could not be written — free some disk space')).toBe(true)
  expect(isDiskFullErrorMessage('This is often a full disk — free some space')).toBe(true)
  expect(isDiskFullErrorMessage('session storage could not be written: permission denied')).toBe(false)
  expect(isDiskFullErrorMessage('network timeout')).toBe(false)

  notifyError(new Error('OSError: [Errno 28] No space left on device: state.db'), 'Prompt failed')

  expect(lastMessage()).toMatch(/Disk full/i)
  expect(lastMessage()).toMatch(/free some space/i)
})

test('in-app toast master switch suppresses notify() and keeps ids stable', () => {
  setInAppToastsEnabled(false)

  expect($inAppToastsEnabled.get()).toBe(false)

  const id = notify({ id: 'toast-a', kind: 'error', message: 'hidden' })

  expect(id).toBe('toast-a')
  expect($notifications.get()).toHaveLength(0)

  // Re-enabling does not resurrect suppressed toasts…
  setInAppToastsEnabled(true)
  expect($notifications.get()).toHaveLength(0)

  // …but new ones show again, and a later dismiss of a suppressed id is a no-op.
  notify({ id: 'toast-b', kind: 'info', message: 'visible' })
  expect($notifications.get().map(n => n.id)).toEqual(['toast-b'])
})

test('gateway error toast id keys on the HTTP status, not the message text', () => {
  // A retrying provider varies its message per attempt; these must collapse
  // into one toast instead of stacking one per failure.
  const first = gatewayErrorToastId('429 Rate limit exceeded, retry after 37s')
  const second = gatewayErrorToastId('429 Rate limit exceeded, retry after 19s')

  expect(first).toBe(second)
  expect(first).toBe('gateway-error:429')

  // Different status → a distinct toast.
  expect(gatewayErrorToastId('503 upstream unavailable')).not.toBe(first)

  // No status in the text → dedupe on the first line.
  const noStatus1 = gatewayErrorToastId('minimax/minimax-m3-free failed\nretry-after: 12s')
  const noStatus2 = gatewayErrorToastId('minimax/minimax-m3-free failed\nretry-after: 45s')

  expect(noStatus1).toBe(noStatus2)
})

test('session storage write failure is treated as disk-full class', () => {
  notifyError(
    new Error('disk full: session storage could not be written — free some disk space and try again'),
    'Prompt failed'
  )

  expect(lastMessage()).toMatch(/Disk full/i)
})

test('code-skew 503 unwraps to a restart-required summary, not raw IPC JSON', () => {
  notifyError(
    new Error(
      'Error invoking remote method \'hermes:api\': Error: 503: {"detail":"Restart required: This process is running code from 08b4875f4a but the checkout on disk is now 48d2528066."}'
    ),
    'Could not load models'
  )

  expect(lastMessage()).toMatch(/running old code after an update/i)
  expect(lastMessage()).not.toMatch(/hermes:api/)
  expect(lastMessage()).not.toMatch(/systemctl/)
})
