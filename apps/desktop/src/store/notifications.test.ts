import { beforeEach, expect, test } from 'vitest'

import { clearAgentNotice, showAgentNotice } from './agent-notices'
import {
  $notifications,
  clearNotifications,
  dismissNotification,
  isDiskFullErrorMessage,
  notify,
  notifyError
} from './notifications'

test('a sticky agent warning survives routine notification traffic until explicitly cleared', () => {
  showAgentNotice({
    key: 'context-maintenance:profile:session',
    text: 'Compaction failed twice',
    kind: 'sticky',
    level: 'warn'
  })

  for (let index = 0; index < 6; index += 1) {
    notify({ id: `routine-${index}`, message: 'Saved', kind: 'success' })
  }

  expect($notifications.get().filter(item => item.id === 'context-maintenance:profile:session')).toHaveLength(1)
  expect($notifications.get().filter(item => item.id.startsWith('routine-'))).toHaveLength(4)
  dismissNotification('context-maintenance:profile:session')
  expect($notifications.get().some(item => item.id === 'context-maintenance:profile:session')).toBe(false)
})

test('context notices with matching remote database paths remain isolated by connection', () => {
  const payload = {
    key: 'context-maintenance:profile:session',
    text: 'Compaction failed',
    kind: 'sticky',
    level: 'warn'
  }

  showAgentNotice(payload, 'connection-a')
  showAgentNotice({ ...payload, text: 'Second connection' }, 'connection-b')
  expect($notifications.get()).toHaveLength(2)

  clearAgentNotice(payload.key, 'connection-a')
  expect($notifications.get()).toHaveLength(1)
  expect($notifications.get()[0].message).toBe('Second connection')
})

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
