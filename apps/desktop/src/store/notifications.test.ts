import { afterEach, describe, expect, it } from 'vitest'

import { setRuntimeI18nLocale } from '@/i18n'

import { $notifications, clearNotifications, notifyError } from './notifications'

describe('notification error localization', () => {
  afterEach(() => {
    clearNotifications()
    setRuntimeI18nLocale('en')
  })

  it('summarizes backend timeouts in Arabic without exposing the English transport error', () => {
    setRuntimeI18nLocale('ar')

    notifyError(new Error('Timed out connecting to Hermes backend after 15000ms'), 'تعذر تحميل الحالة')

    expect($notifications.get()[0]).toMatchObject({
      detail: undefined,
      message: 'لم يستجب هرمس خلال ١٥ ثانية. تحقق من الاتصال ثم أعد المحاولة.',
      title: 'تعذر تحميل الحالة'
    })
  })
})
