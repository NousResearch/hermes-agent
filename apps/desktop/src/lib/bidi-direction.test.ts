import { describe, expect, it } from 'vitest'

import { resolveTextDirection } from './bidi-direction'

describe('resolveTextDirection', () => {
  it('ignores leading neutral punctuation before RTL text', () => {
    expect(resolveTextDirection('- شغّل Hermes')).toBe('rtl')
    expect(resolveTextDirection('؟ شغّل Hermes')).toBe('rtl')
    expect(resolveTextDirection('(شغّل Hermes)')).toBe('rtl')
    expect(resolveTextDirection('2026: شغّل Hermes')).toBe('rtl')
  })

  it('lets RTL text after leading code-like tokens own the sentence direction', () => {
    expect(resolveTextDirection('`npm test` شغّل الأول')).toBe('rtl')
    expect(resolveTextDirection('@file:`apps/desktop/a.ts` شوف الملف')).toBe('rtl')
    expect(resolveTextDirection('./run.sh شغّل السكريبت')).toBe('rtl')
    expect(resolveTextDirection('/some-skill شغّل ده')).toBe('rtl')
  })

  it('falls back to the leading strong LTR text when there is no RTL sentence body', () => {
    expect(resolveTextDirection('run tests الأول')).toBe('ltr')
    expect(resolveTextDirection('`npm test`')).toBe('ltr')
    expect(resolveTextDirection('@file:`apps/desktop/a.ts`')).toBe('ltr')
  })

  it('uses the dominant sentence script when an English brand starts Arabic prose', () => {
    expect(resolveTextDirection('Alibaba نزلت Qwen3.8-Max والمقلب الحلو إنك بتكلم الخير ده')).toBe('rtl')
    expect(resolveTextDirection('DeepSeek نزلت V4 beta شغالة على الأسعار الصينية')).toBe('rtl')
    expect(resolveTextDirection('Moonshot (Kimi) نزلوا K3 وفتحوا الـ infrastructure بتاعهم')).toBe('rtl')
    expect(resolveTextDirection('Google عندها Gemini 3.5 + Gemini Omni + computer use في Flash')).toBe('rtl')
  })

  it('keeps leading neutral punctuation outside an English-brand Arabic sentence', () => {
    expect(resolveTextDirection('• Google عندها Gemini 3.5')).toBe('rtl')
    expect(resolveTextDirection('— OpenAI لسه مكملة بـ GPT-5.6')).toBe('rtl')
  })
})
