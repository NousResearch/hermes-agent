import { describe, expect, it } from 'vitest'

import { applicationMenuCopy, normalizeApplicationMenuLocale } from './application-menu-copy'

describe('application menu localization', () => {
  it('normalizes Arabic locale aliases', () => {
    expect(normalizeApplicationMenuLocale('ar')).toBe('ar')
    expect(normalizeApplicationMenuLocale('ar-EG')).toBe('ar')
    expect(normalizeApplicationMenuLocale('en')).toBe('en')
    expect(normalizeApplicationMenuLocale(null)).toBe('en')
  })

  it('provides a complete Arabic native menu', () => {
    const english = applicationMenuCopy('en')
    const arabic = applicationMenuCopy('ar')

    expect(Object.keys(arabic)).toEqual(Object.keys(english))
    expect(arabic.file).toBe('ملف')
    expect(arabic.checkForUpdates).toBe('التحقق من التحديثات…')
  })
})
