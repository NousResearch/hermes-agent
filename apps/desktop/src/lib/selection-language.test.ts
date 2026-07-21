import { describe, expect, it } from 'vitest'

import {
  detectSelectionLanguage,
  resolveTranslateTarget
} from './selection-language'

describe('selection language auto direction', () => {
  it('defaults English-dominant text to Arabic', () => {
    expect(detectSelectionLanguage('Hello Billy, what do you want to work on?')).toBe('en')
    expect(resolveTranslateTarget('Hello Billy, what do you want to work on?')).toBe('ar')
  })

  it('defaults Arabic-dominant text to English', () => {
    const arabic = 'مرحباً بيلي، بماذا تريد أن تعمل؟'
    expect(detectSelectionLanguage(arabic)).toBe('ar')
    expect(resolveTranslateTarget(arabic)).toBe('en')
  })

  it('uses dominant script on mixed text', () => {
    expect(resolveTranslateTarget('Hello مرحبا world again more english words')).toBe('ar')
    expect(resolveTranslateTarget('مرحبا بكم في هذا النص العربي الطويل Hello')).toBe('en')
  })

  it('honours explicit target override', () => {
    expect(resolveTranslateTarget('Hello', 'en')).toBe('en')
    expect(resolveTranslateTarget('مرحبا', 'ar')).toBe('ar')
    expect(resolveTranslateTarget('Hello', 'auto')).toBe('ar')
  })

  it('treats empty/non-script text as English-source → Arabic target', () => {
    expect(detectSelectionLanguage('123 !!!')).toBe('en')
    expect(resolveTranslateTarget('123 !!!')).toBe('ar')
  })
})
