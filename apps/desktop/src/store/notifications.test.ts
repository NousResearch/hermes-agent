import { beforeEach, expect, test, vi } from 'vitest'

import { $notifications, clearNotifications, isDiskFullErrorMessage, notifyError } from './notifications'

beforeEach(() => {
  clearNotifications()
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

test('session storage write failure is treated as disk-full class', () => {
  notifyError(
    new Error('disk full: session storage could not be written — free some disk space and try again'),
    'Prompt failed'
  )

  expect(lastMessage()).toMatch(/Disk full/i)
})

test('notifyError posts the full error to desktop.log, not the summary', () => {
  const logLine = vi.fn()

  const previous = (window as unknown as { hermesDesktop?: unknown }).hermesDesktop

  ;(window as unknown as { hermesDesktop: { logLine: typeof logLine } }).hermesDesktop = { logLine }

  try {
    const error = new Error('sqlite3.OperationalError: database is locked')
    error.stack = 'Error: sqlite3.OperationalError: database is locked\n    at saveSession (session.ts:12)'

    notifyError(error, 'Prompt failed')

    expect(logLine).toHaveBeenCalledTimes(1)
    expect(logLine.mock.calls[0][0]).toContain('Prompt failed')
    expect(logLine.mock.calls[0][0]).toContain('database is locked')
    expect(logLine.mock.calls[0][0]).toContain('session.ts:12')
  } finally {
    if (previous === undefined) {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    } else {
      ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = previous
    }
  }
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
