// Both Settings actions that can ask for browser notification permission must
// leave the permission row showing the answer, not the snapshot it mounted with.
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { NotificationsSettings } from './notifications-settings'

vi.mock('@/lib/host-capabilities', () => ({ hostHas: (name: string) => name === 'browserNotifications' }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))
vi.mock('@/store/native-notifications', async () => {
  const { atom } = await import('nanostores')

  return {
    $nativeNotifyPrefs: atom({ enabled: true, kinds: {} }),
    NATIVE_NOTIFICATION_KINDS: [],
    sendTestNativeNotification: vi.fn(async () => true),
    setNativeNotifyEnabled: vi.fn(),
    setNativeNotifyKind: vi.fn()
  }
})

class FakeNotification {
  static answer: NotificationPermission = 'granted'
  static permission: NotificationPermission = 'default'
  static requestPermission = vi.fn(async () => {
    FakeNotification.permission = FakeNotification.answer

    return FakeNotification.permission
  })
}

beforeEach(() => {
  FakeNotification.permission = 'default'
  vi.stubGlobal('Notification', FakeNotification)
  vi.stubGlobal('isSecureContext', true)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function renderAlerts() {
  render(
    <MemoryRouter>
      <NotificationsSettings subpage="alerts" />
    </MemoryRouter>
  )
}

it.each([
  ['granted', /allowed/i],
  ['denied', /blocked/i]
] as const)('the permission row follows an answer given through Send test notification (%s)', async (answer, state) => {
  FakeNotification.answer = answer
  renderAlerts()
  expect(screen.getByRole('button', { name: 'Allow' })).toBeTruthy()

  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Send test notification' }))
  })

  expect(FakeNotification.requestPermission).toHaveBeenCalled()
  expect(screen.queryByRole('button', { name: 'Allow' })).toBeNull()
  expect(screen.getByText(state)).toBeTruthy()
})
