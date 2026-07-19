import { afterEach, describe, expect, it } from 'vitest'

import { setRuntimeI18nLocale } from '@/i18n'

import { $notifications, clearNotifications, notifyError } from './notifications'

const BACKEND_TIMEOUT = 'Timed out connecting to Hermes backend after 15000ms'

describe('notification error reporting', () => {
  afterEach(() => {
    clearNotifications()
    setRuntimeI18nLocale('en')
  })

  // The localization layer must never swallow a diagnostic an English user
  // could read before it existed. Transport errors have no catalog entry on
  // purpose: they pass through verbatim so the text stays searchable.
  it('keeps the transport error text visible instead of summarizing it away', () => {
    notifyError(new Error(BACKEND_TIMEOUT), 'Could not load status')

    expect($notifications.get()[0]).toMatchObject({
      message: BACKEND_TIMEOUT,
      title: 'Could not load status'
    })
  })

  it('keeps the same transport detail visible in Arabic', () => {
    setRuntimeI18nLocale('ar')

    notifyError(new Error(BACKEND_TIMEOUT), 'تعذر تحميل الحالة')

    const shown = $notifications.get()[0]

    expect(shown?.title).toBe('تعذر تحميل الحالة')
    expect(`${shown?.message} ${shown?.detail ?? ''}`).toContain(BACKEND_TIMEOUT)
  })

  // Rules that *did* exist upstream still summarize, and still keep the raw
  // provider text as the expandable detail line.
  it('summarizes a known provider error without dropping its raw text', () => {
    notifyError(new Error('Incorrect API key provided: sk-123. Error code: 401'), 'Could not save key')

    const shown = $notifications.get()[0]

    expect(shown?.message).toBe('OpenAI rejected the API key (401 invalid_api_key).')
    expect(shown?.detail).toContain('Incorrect API key provided')
  })
})
