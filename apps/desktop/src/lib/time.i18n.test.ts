import { afterEach, describe, expect, it } from 'vitest'

import { setRuntimeI18nLocale } from '@/i18n/runtime'

import { fmtDate, fmtDayTime, fmtMonth, relativeTime } from './time'

// The app's language is the one picked in Settings, not the OS regional locale
// Chromium hands `Intl` for `undefined`. With `undefined`, a Chinese UI on an
// en-US region rendered English month dividers ("AUGUST") and English ages
// ("in 10 hr.") beside Chinese labels. These pin the contract: the active app
// language — not the renderer default — decides the format, and an
// already-created formatter re-resolves when that language changes.
//
// Assertions stay on the contract (which language, which script) rather than on
// one ICU spelling: `month: 'long'` is 八月 under zh-CN today, and a glyph-level
// snapshot would only break on an ICU bump.
const AUGUST_16_2026 = new Date(2026, 7, 16, 14, 30)

afterEach(() => {
  setRuntimeI18nLocale('en')
})

describe('time formatting follows the app language', () => {
  it('names months in the active language, not the renderer default', () => {
    setRuntimeI18nLocale('zh')
    const zhMonth = fmtMonth.format(AUGUST_16_2026)

    expect(zhMonth).toMatch(/月$/)
    expect(zhMonth).not.toMatch(/august/i)

    setRuntimeI18nLocale('en')
    expect(fmtMonth.format(AUGUST_16_2026)).toBe('August')
  })

  it('re-resolves an already-created formatter after a language switch', () => {
    setRuntimeI18nLocale('en')
    const before = fmtDayTime.format(AUGUST_16_2026)

    setRuntimeI18nLocale('zh')
    const after = fmtDayTime.format(AUGUST_16_2026)

    expect(before).toMatch(/aug/i)
    expect(after).toContain('月')
    expect(after).not.toMatch(/aug/i)
  })

  it('formats a full date without falling back to the renderer locale', () => {
    setRuntimeI18nLocale('zh')
    const zhDate = fmtDate.format(AUGUST_16_2026)

    expect(zhDate).toContain('2026')
    expect(zhDate).toContain('月')
    expect(zhDate).not.toMatch(/aug/i)
  })

  it('renders a future age in the active language', () => {
    const inTenHours = Date.now() + 10 * 60 * 60 * 1000

    setRuntimeI18nLocale('zh')
    expect(relativeTime(inTenHours)).toContain('小时')

    setRuntimeI18nLocale('en')
    expect(relativeTime(inTenHours)).toMatch(/hr|hour/)
  })
})
