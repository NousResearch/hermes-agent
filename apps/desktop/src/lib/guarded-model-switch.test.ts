import { afterEach, expect, it, vi } from 'vitest'

import { setRuntimeI18nLocale } from '@/i18n/runtime'
import { confirm } from '@/store/confirm'
import { notify, notifyError } from '@/store/notifications'

import { surfaceModelSwitchConfirm } from './guarded-model-switch'

vi.mock('@/store/confirm', () => ({ confirm: vi.fn(async () => true) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
afterEach(() => {
  vi.clearAllMocks()
  setRuntimeI18nLocale('en')
})

it('keeps the translated explanation and verbatim cost/data warning in the confirmation dialog', async () => {
  setRuntimeI18nLocale('ko')

  for (const warning of [
    'Cost: $10 per million input tokens. Model: provider/raw-ID.',
    'This contributor model trains on your data.',
    undefined
  ]) {
    const requestConfirmed = vi.fn(async () => ({}))
    expect(
      await surfaceModelSwitchConfirm({ confirmMessage: warning, failureMessage: '모델 변경 실패', requestConfirmed })
    ).toBe(true)
    const dialog = vi.mocked(confirm).mock.lastCall![0]
    expect(dialog.description).toBeTruthy()

    if (warning) {
      expect(dialog.description).toContain(warning)
    }

    expect(requestConfirmed).toHaveBeenCalledTimes(1)
  }
})

it('discards stale confirmations and refuses repeated confirmation without looping', async () => {
  const requestConfirmed = vi.fn(async () => ({ confirm_required: true, confirm_message: 'Original repeated warning' }))

  const rollback = vi.fn(),
    finish = vi.fn()

  for (const stale of [true, false]) {
    expect(
      await surfaceModelSwitchConfirm({
        failureMessage: 'Failed',
        requestConfirmed,
        rollback,
        finish,
        isStale: () => stale
      })
    ).toBe(false)
    expect(requestConfirmed).toHaveBeenCalledTimes(stale ? 0 : 1)
  }

  expect(notify).toHaveBeenCalledTimes(1)
  expect(rollback).toHaveBeenCalledTimes(1)
  expect(finish).not.toHaveBeenCalled()
  expect(notifyError).toHaveBeenCalledWith(expect.objectContaining({ message: 'Original repeated warning' }), 'Failed')
})
