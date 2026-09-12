import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { SECTIONS } from '@/app/settings/constants'
import { fieldCopyForSchemaKey } from '@/app/settings/field-copy'

import { TRANSLATIONS } from './catalog'
import { en } from './en'
import { setRuntimeI18nLocale, translateNow } from './runtime'
import { zh } from './zh'

describe('desktop i18n runtime translator', () => {
  beforeEach(() => {
    setRuntimeI18nLocale('en')
  })

  afterEach(() => {
    setRuntimeI18nLocale('en')
  })

  it('translates string paths for the active runtime locale', () => {
    setRuntimeI18nLocale('zh')

    expect(translateNow('boot.ready')).toBe('Hermes 桌面版已就绪')
    expect(translateNow('notifications.voice.noSpeechDetected')).toBe('没有检测到语音')
    expect(translateNow('composer.lookupNoMatches')).toBe('没有匹配项。')
    expect(translateNow('assistant.tool.statusRecovered')).toBe('已恢复')
  })

  it('passes arguments to function translations', () => {
    expect(translateNow('notifications.updateReadyMessage', 2)).toBe('2 new changes available.')
  })

  it('translates migrated overlap keys for newly supported locales', () => {
    setRuntimeI18nLocale('ja')
    expect(translateNow('common.save')).toBe('保存')

    setRuntimeI18nLocale('zh-hant')
    expect(translateNow('cron.promptPlaceholder')).toBe('代理每次執行時應做什麼？')
  })

  it('translates settings copy for newly supported locales', () => {
    setRuntimeI18nLocale('ja')
    expect(translateNow('settings.appearance.title')).toBe('外観')
    expect(translateNow('settings.nav.providers')).toBe('プロバイダー')

    setRuntimeI18nLocale('zh-hant')
    expect(translateNow('settings.appearance.title')).toBe('外觀')
    expect(translateNow('settings.nav.providerApiKeys')).toBe('API 金鑰')

    setRuntimeI18nLocale('ar')
    expect(translateNow('settings.appearance.reasoningCollapsedTitle')).toBe('طي التفكير افتراضيًا')
    expect(translateNow('settings.appearance.reasoningCollapsedDesc')).toBe(
      'أبقِ التفكير المتدفق متاحًا دون توسيعه حتى تفتحه.'
    )
  })

  it('keeps translated settings field copy addressable from schema keys', () => {
    const field = ['display', 'show_reasoning'].join('.')

    expect(fieldCopyForSchemaKey(zh.settings.fieldLabels, field)).toBe('推理过程块')
    expect(fieldCopyForSchemaKey(zh.settings.fieldDescriptions, field)).toBe('当后端提供推理内容时予以显示。')
  })

  // Contract shape: the zh overlay supplies a real translation (non-empty,
  // contains CJK) instead of falling through to the English source. Assumes
  // the intended translation never equals the English text — don't reuse
  // this shape for keys where zh legitimately mirrors en (brand names etc.).
  const CJK = /[\u4e00-\u9fff]/

  it('localizes the browser real-profile setting instead of falling back to English', () => {
    const field = 'browser.use_real_profile'

    const label = fieldCopyForSchemaKey(zh.settings.fieldLabels, field)
    const description = fieldCopyForSchemaKey(zh.settings.fieldDescriptions, field)

    expect(label).toBeTruthy()
    expect(label).not.toBe(fieldCopyForSchemaKey(en.settings.fieldLabels, field))
    expect(label).toMatch(CJK)
    expect(description).toBeTruthy()
    expect(description).not.toBe(fieldCopyForSchemaKey(en.settings.fieldDescriptions, field))
    expect(description).toMatch(CJK)
  })

  it('localizes the browser settings section title', () => {
    const englishLabel = SECTIONS.find(section => section.id === 'browser')?.label
    const title = zh.settings.sections.browser

    expect(title).toBeTruthy()
    expect(title).not.toBe(englishLabel)
    expect(title).toMatch(CJK)
  })

  it('falls back to English when the active locale cannot resolve a key', () => {
    const boot = TRANSLATIONS.ja.boot as { ready?: string }
    const originalReady = boot.ready

    try {
      boot.ready = undefined
      setRuntimeI18nLocale('ja')

      expect(translateNow('boot.ready')).toBe('Hermes Desktop is ready')
    } finally {
      boot.ready = originalReady
    }
  })

  it('returns the key when no locale can resolve a path', () => {
    setRuntimeI18nLocale('zh')

    expect(translateNow('missing.path')).toBe('missing.path')
  })
})
