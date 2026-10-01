import { beforeEach, expect, test } from 'vitest'

import { en } from '@/i18n/en'

import { $notifications, clearNotifications, isDiskFullErrorMessage, notifyError } from './notifications'
import { $backendRestartRequest, $routeRequest } from './recovery-requests'

beforeEach(() => {
  clearNotifications()
})

function lastMessage(): string {
  return $notifications.get()[0]?.message ?? ''
}

// Regression for #39365: a gateway auth 401 (bad API_SERVER_KEY) must not be
// summarized as a provider (OpenAI/OpenRouter) API key problem. The toast says
// "sign in again" in plain words and opens Gateways — no env-var names.
test('gateway_auth_failed error is summarized as sign-in, with an Open Gateways action', () => {
  notifyError(
    new Error(
      '401 {"error": {"message": "Invalid gateway API key (API_SERVER_KEY)", "type": "gateway_auth_error", "code": "gateway_auth_failed"}}'
    ),
    'Request failed'
  )

  expect(lastMessage()).not.toMatch(/API_SERVER_KEY|OpenAI|authentication failed/i)

  const action = $notifications.get()[0]?.action
  expect(action?.label).toBe(en.notifications.actions.openGateways)
  action?.onClick()
  expect($routeRequest.get()?.path).toBe('/settings?tab=gateway')
})

test('provider invalid_api_key error maps to the OpenAI summary and deep-links to Keys', () => {
  notifyError(
    new Error('401 {"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}'),
    'Request failed'
  )

  expect(lastMessage()).not.toMatch(/401|invalid_api_key/)
  $notifications.get()[0]?.action?.onClick()
  expect($routeRequest.get()?.path).toBe('/settings?tab=keys&key=OPENAI_API_KEY')
})

test('ELEVENLABS_API_KEY not set toasts plain copy with an Open Keys action for that key', () => {
  notifyError(new Error('ELEVENLABS_API_KEY not set'), 'Voice failed')

  expect(lastMessage()).not.toMatch(/ELEVENLABS_API_KEY|STT/)
  $notifications.get()[0]?.action?.onClick()
  expect($routeRequest.get()?.path).toBe('/settings?tab=keys&key=ELEVENLABS_API_KEY')
})

test('structured storage_* error codes route to Maintenance', () => {
  notifyError(new Error('500 {"detail":{"message":"database is locked","code":"storage_locked"}}'), 'Prompt failed')

  $notifications.get()[0]?.action?.onClick()
  expect($routeRequest.get()?.path).toBe('/command-center?section=maintenance')
})

test('405 method-not-allowed toasts a restart in plain words with a Restart Hermes action', () => {
  const before = $backendRestartRequest.get()
  notifyError(new Error('405 Method Not Allowed'), 'Request failed')

  expect(lastMessage()).not.toMatch(/405|Method Not Allowed|backend/i)
  expect($notifications.get()[0]?.action?.label).toBe(en.notifications.actions.restartHermes)
  $notifications.get()[0]?.action?.onClick()
  expect($backendRestartRequest.get()).toBe(before + 1)
})

test('disk-full / ENOSPC phrasings are classified as disk-full, other storage failures are not', () => {
  expect(isDiskFullErrorMessage('OSError: [Errno 28] No space left on device')).toBe(true)
  expect(isDiskFullErrorMessage('sqlite3.OperationalError: database or disk is full')).toBe(true)
  expect(isDiskFullErrorMessage('disk full: session storage could not be written — free some disk space')).toBe(true)
  expect(isDiskFullErrorMessage('This is often a full disk — free some space')).toBe(true)
  expect(isDiskFullErrorMessage('session storage could not be written: permission denied')).toBe(false)
  expect(isDiskFullErrorMessage('network timeout')).toBe(false)
})

test('code-skew 503 unwraps to a restart-required summary, not raw IPC JSON', () => {
  notifyError(
    new Error(
      'Error invoking remote method \'hermes:api\': Error: 503: {"detail":"Restart required: This process is running code from 08b4875f4a but the checkout on disk is now 48d2528066."}'
    ),
    'Could not load models'
  )

  expect(lastMessage()).not.toMatch(/hermes:api|systemctl|backend/i)
  const before = $backendRestartRequest.get()
  expect($notifications.get()[0]?.action?.label).toBe(en.notifications.actions.restartHermes)
  $notifications.get()[0]?.action?.onClick()
  expect($backendRestartRequest.get()).toBe(before + 1)
})
