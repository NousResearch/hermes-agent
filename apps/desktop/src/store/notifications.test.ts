import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'

import { en } from '@/i18n/en'

import {
  $notifications,
  approvalNoticeId,
  clearNotifications,
  dismissApprovalNotice,
  isDiskFullErrorMessage,
  notify,
  notifyError
} from './notifications'
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

test.each(['missing', 'closed'] as const)(
  'notifyError still shows a toast when the log bridge is %s',
  (state: 'missing' | 'closed'): void => {
    vi.stubGlobal(
      'hermesDesktop',
      state === 'missing'
        ? undefined
        : {
            logLine: (): never => {
              throw new Error('IPC channel closed')
            }
          }
    )

    try {
      const id: string = notifyError(new Error('database is locked'), 'Prompt failed')

      expect($notifications.get()[0]).toMatchObject({ id, kind: 'error', message: 'database is locked' })
    } finally {
      vi.unstubAllGlobals()
    }
  }
)

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

describe('pinned approval notices', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  test('a pinned notice ignores its duration: it stays until explicitly dismissed', () => {
    vi.useFakeTimers()
    notify({ id: 'approval:s1:r1', kind: 'warning', message: 'run ls', pinned: true, durationMs: 40 })

    vi.advanceTimersByTime(5_000)
    expect($notifications.get().some(item => item.id === 'approval:s1:r1')).toBe(true)

    clearNotifications()
    expect($notifications.get().some(item => item.id === 'approval:s1:r1')).toBe(false)
  })

  test('a re-raised approval replaces its notice instead of stacking a duplicate', () => {
    notify({ id: approvalNoticeId('s1', 'r1'), kind: 'warning', message: 'first', pinned: true })
    notify({ id: approvalNoticeId('s1', 'r1'), kind: 'warning', message: 'second', pinned: true })

    const matching = $notifications.get().filter(item => item.id === approvalNoticeId('s1', 'r1'))
    expect(matching).toHaveLength(1)
    expect(matching[0]?.message).toBe('second')
  })

  test('approvalNoticeId keys on the session and the queue request id', () => {
    expect(approvalNoticeId('s1', 'r1')).toBe('approval:s1:r1')
    expect(approvalNoticeId('s1')).toBe('approval:s1')
    expect(approvalNoticeId(null)).toBe('approval:')
  })

  test('dismissApprovalNotice removes exactly one request notice', () => {
    notify({ id: approvalNoticeId('s1', 'r1'), kind: 'warning', message: 'a', pinned: true })
    notify({ id: approvalNoticeId('s1', 'r2'), kind: 'warning', message: 'b', pinned: true })

    dismissApprovalNotice('s1', 'r1')
    expect($notifications.get().some(item => item.id === approvalNoticeId('s1', 'r1'))).toBe(false)
    expect($notifications.get().some(item => item.id === approvalNoticeId('s1', 'r2'))).toBe(true)
  })

  test('dismissApprovalNotice without a request id clears the whole session, not other sessions', () => {
    notify({ id: approvalNoticeId('s1', 'r1'), kind: 'warning', message: 'a', pinned: true })
    notify({ id: approvalNoticeId('s1', 'r2'), kind: 'warning', message: 'b', pinned: true })
    notify({ id: approvalNoticeId('s2', 'r1'), kind: 'warning', message: 'c', pinned: true })

    dismissApprovalNotice('s1')
    expect($notifications.get().some(item => item.id.startsWith('approval:s1'))).toBe(false)
    expect($notifications.get().some(item => item.id === approvalNoticeId('s2', 'r1'))).toBe(true)
  })

  test('dismissApprovalNotice without a session hint clears every approval notice, keeps unrelated toasts', () => {
    notify({ id: approvalNoticeId('s1', 'r1'), kind: 'warning', message: 'a', pinned: true })
    notify({ id: approvalNoticeId('s2', 'r2'), kind: 'warning', message: 'b', pinned: true })
    notify({ id: 'unrelated', kind: 'info', message: 'saved', durationMs: 5_000 })

    dismissApprovalNotice()
    expect($notifications.get().some(item => item.id.startsWith('approval:'))).toBe(false)
    expect($notifications.get().some(item => item.id === 'unrelated')).toBe(true)
  })
})
