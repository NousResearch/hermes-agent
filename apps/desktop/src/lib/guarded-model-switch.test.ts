import { afterEach, expect, it, vi } from 'vitest'

import { setRuntimeI18nLocale } from '@/i18n/runtime'
import { dismissNotification, notify, notifyError } from '@/store/notifications'

import { surfaceModelSwitchConfirm } from './guarded-model-switch'

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(() => 'warning-id'),
  dismissNotification: vi.fn(),
  notifyError: vi.fn()
}))
afterEach(() => {
  vi.clearAllMocks()
  setRuntimeI18nLocale('en')
})

it('adds a Korean summary without removing backend cost or data-use warnings, including the no-message path', async () => {
  setRuntimeI18nLocale('ko')

  for (const warning of [
    'Cost: $10 per million input tokens. Model: provider/raw-ID.',
    'This contributor model trains on your data.',
    undefined
  ]) {
    const requestConfirmed = vi.fn(async () => ({}))
    surfaceModelSwitchConfirm({
      confirmLabel: '확인',
      confirmMessage: warning,
      failureMessage: '모델 변경 실패',
      requestConfirmed
    })
    const notification = vi.mocked(notify).mock.lastCall![0]
    expect(notification.message).toContain('비용 및 데이터 사용 조건')

    if (warning) {
      expect(notification.message).toContain(warning)
    }

    expect(requestConfirmed).not.toHaveBeenCalled()
    await notification.action!.onClick()
    expect(requestConfirmed).toHaveBeenCalledTimes(1)
  }
})

it('still discards stale confirmations and refuses a second confirmation request without looping', async () => {
  const requestConfirmed = vi.fn(async () => ({ confirm_required: true, confirm_message: 'Original repeated warning' }))

  const rollback = vi.fn(),
    finish = vi.fn()

  for (const stale of [true, false]) {
    surfaceModelSwitchConfirm({
      confirmLabel: 'Confirm',
      failureMessage: 'Failed',
      requestConfirmed,
      rollback,
      finish,
      isStale: () => stale
    })
    await vi.mocked(notify).mock.lastCall![0].action!.onClick()
    expect(requestConfirmed).toHaveBeenCalledTimes(stale ? 0 : 1)
  }

  expect(dismissNotification).toHaveBeenCalledWith('warning-id')
  expect(rollback).toHaveBeenCalledTimes(1)
  expect(finish).not.toHaveBeenCalled()
  expect(notifyError).toHaveBeenCalledWith(expect.objectContaining({ message: 'Original repeated warning' }), 'Failed')
})
