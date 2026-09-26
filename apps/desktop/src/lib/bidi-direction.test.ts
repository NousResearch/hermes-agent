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

  it('treats English technical labels as neutral prefixes to Arabic notes', () => {
    expect(resolveTextDirection('Task 16 Step1 اهو الكلام مش هيتقفل بكلمة green')).toBe('rtl')
    expect(resolveTextDirection('Net-complexity gate مش بس إن أسماء tables/workers لازم تثبت')).toBe('rtl')
    expect(resolveTextDirection('Radar watermark: failure processed لا تتعلم.')).toBe('rtl')
    expect(resolveTextDirection('Learning terminal existing: NOOP مايتحولش.')).toBe('rtl')
    expect(resolveTextDirection('Coach hypothesis job failures: مايتجنوش.')).toBe('rtl')
  })

  it('keeps an English sentence containing one Arabic word LTR', () => {
    expect(resolveTextDirection('Can you explain what مرحبا means?')).toBe('ltr')
    expect(resolveTextDirection('Please translate this into Arabic مرحبا for me')).toBe('ltr')
    expect(resolveTextDirection('The error message says مرحبا in the log')).toBe('ltr')
    expect(resolveTextDirection('Why does مرحبا appear in this test?')).toBe('ltr')
    expect(resolveTextDirection('I think مرحبا is wrong here')).toBe('ltr')
    expect(resolveTextDirection('Use مرحبا as the greeting')).toBe('ltr')
    expect(resolveTextDirection('This is the مرحبا example from the docs')).toBe('ltr')
    expect(resolveTextDirection('Is مرحبا the right word for hello?')).toBe('ltr')
    expect(resolveTextDirection('Add a test for the مرحبا case')).toBe('ltr')
    expect(resolveTextDirection('Run tests الأول')).toBe('ltr')
    // Casing must not decide: the lowercased controls stay LTR as well.
    expect(resolveTextDirection('can you explain what مرحبا means?')).toBe('ltr')
    expect(resolveTextDirection('the error message says مرحبا in the log')).toBe('ltr')
  })

  it('keeps a bare English lead with a trailing Arabic word LTR', () => {
    expect(resolveTextDirection('Try مرحبا')).toBe('ltr')
    expect(resolveTextDirection('Say مرحبا')).toBe('ltr')
    expect(resolveTextDirection('Send مرحبا!')).toBe('ltr')
    expect(resolveTextDirection('Try مرحبا?')).toBe('ltr')
    expect(resolveTextDirection('Use مرحبا')).toBe('ltr')
    expect(resolveTextDirection('Thanks مرحبا')).toBe('ltr')
    expect(resolveTextDirection('The مرحبا.')).toBe('ltr')
    expect(resolveTextDirection('Try مرحبا in the app')).toBe('ltr')
    expect(resolveTextDirection('Google عندها Gemini 3.5')).toBe('rtl')
  })

  it('keeps leading neutral punctuation outside an English-brand Arabic sentence', () => {
    expect(resolveTextDirection('• Google عندها Gemini 3.5')).toBe('rtl')
    expect(resolveTextDirection('— OpenAI لسه مكملة بـ GPT-5.6')).toBe('rtl')
  })
})
