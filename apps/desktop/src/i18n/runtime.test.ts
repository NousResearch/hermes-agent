import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { fieldCopyForSchemaKey } from '@/app/settings/field-copy'

import { TRANSLATIONS } from './catalog'
import { en } from './en'
import { ru } from './ru'
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
  })

  it('keeps translated settings field copy addressable from schema keys', () => {
    const field = ['display', 'show_reasoning'].join('.')

    expect(fieldCopyForSchemaKey(zh.settings.fieldLabels, field)).toBe('推理过程块')
    expect(fieldCopyForSchemaKey(zh.settings.fieldDescriptions, field)).toBe('当后端提供推理内容时予以显示。')
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

  it('resolves keys against the active Russian locale', () => {
    setRuntimeI18nLocale('ru')

    // Compare against the locale objects rather than pinning wording: the value
    // must be Russian's own translation and distinct from the English default.
    expect(translateNow('common.save')).toBe(ru.common.save)
    expect(translateNow('common.save')).not.toBe(en.common.save)
  })

  it('falls back to English when the Russian locale cannot resolve a key', () => {
    const tool = TRANSLATIONS.ru.assistant.tool as { statusDone?: string }
    const originalStatusDone = tool.statusDone

    try {
      tool.statusDone = undefined
      setRuntimeI18nLocale('ru')

      expect(translateNow('assistant.tool.statusDone')).toBe(en.assistant.tool.statusDone)
    } finally {
      tool.statusDone = originalStatusDone
    }
  })
})
