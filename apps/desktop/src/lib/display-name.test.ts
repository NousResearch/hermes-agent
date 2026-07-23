import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'

import { displayEntityName } from './display-name'

describe('displayEntityName', () => {
  it('keeps "default" verbatim in English (identity presentation)', () => {
    expect(displayEntityName('default', TRANSLATIONS.en)).toBe('default')
  })

  it('localizes the reserved "default" name in zh', () => {
    const zh = TRANSLATIONS.zh

    expect(displayEntityName('default', zh)).toBe(zh.common.defaultName)
    expect(zh.common.defaultName).not.toBe('default')
  })

  it('resolves a human string for every locale (never an empty label)', () => {
    for (const translations of Object.values(TRANSLATIONS)) {
      expect(displayEntityName('default', translations).length).toBeGreaterThan(0)
    }
  })

  it('passes user-created names through untouched in every locale', () => {
    for (const translations of Object.values(TRANSLATIONS)) {
      expect(displayEntityName('coder', translations)).toBe('coder')
      expect(displayEntityName('默认', translations)).toBe('默认')
      expect(displayEntityName('', translations)).toBe('')
    }
  })
})
