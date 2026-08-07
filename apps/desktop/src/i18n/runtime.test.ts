import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { fieldCopyForSchemaKey } from '@/app/settings/field-copy'

import { TRANSLATIONS } from './catalog'
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

  it('registers and translates the Russian locale', () => {
    expect(TRANSLATIONS.ru).toBe(ru)

    setRuntimeI18nLocale('ru')
    expect(translateNow('boot.ready')).toBe('Hermes Desktop готов')
  })

  it('uses Russian plural forms for function translations', () => {
    setRuntimeI18nLocale('ru')

    const cases = [
      [0, 'Ещё 0 уведомлений'],
      [1, 'Ещё 1 уведомление'],
      [2, 'Ещё 2 уведомления'],
      [4, 'Ещё 4 уведомления'],
      [5, 'Ещё 5 уведомлений'],
      [11, 'Ещё 11 уведомлений'],
      [21, 'Ещё 21 уведомление'],
      [22, 'Ещё 22 уведомления'],
      [25, 'Ещё 25 уведомлений'],
      [111, 'Ещё 111 уведомлений']
    ] as const

    for (const [count, expected] of cases) {
      expect(translateNow('notifications.more', count)).toBe(expected)
    }
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
    expect(fieldCopyForSchemaKey(ru.settings.fieldLabels, field)).toBe('Блоки рассуждений')
    expect(fieldCopyForSchemaKey(ru.settings.fieldDescriptions, field)).toBe(
      'Показывать блоки рассуждений, когда их предоставляет бэкенд.'
    )
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
