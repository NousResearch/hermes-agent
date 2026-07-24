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

  it('translates migrated overlap keys for newly supported locales', () => {
    setRuntimeI18nLocale('ja')
    expect(translateNow('common.save')).toBe('保存')

    setRuntimeI18nLocale('zh-hant')
    expect(translateNow('cron.promptPlaceholder')).toBe('代理每次執行時應做什麼？')
  })

  it('translates Russian strings and applies the 11-14 plural exception', () => {
    setRuntimeI18nLocale('ru')

    expect(translateNow('common.save')).toBe('Сохранить')
    expect(translateNow('notifications.more', 1)).toBe('Ещё 1 уведомление')
    expect(translateNow('notifications.more', 2)).toBe('Ещё 2 уведомления')
    expect(translateNow('notifications.more', 5)).toBe('Ещё 5 уведомлений')
    expect(translateNow('notifications.more', 11)).toBe('Ещё 11 уведомлений')
    expect(translateNow('notifications.more', 21)).toBe('Ещё 21 уведомление')
    expect(translateNow('settings.appearance.uiScaleTitle')).toBe('Масштаб интерфейса')
    expect(translateNow('settings.appearance.embedsTitle')).toBe('Встроенные предпросмотры')
    expect(translateNow('settings.appearance.pet.title')).toBe('Питомец')
    expect(translateNow('settings.appearance.pet.count', 11)).toBe('11 питомцев.')
  })

  it('translates Russian file, memory graph, and schema settings surfaces', () => {
    setRuntimeI18nLocale('ru')

    expect(translateNow('fileMenu.copyPath')).toBe('Копировать путь')
    expect(translateNow('starmap.title')).toBe('Граф памяти')
    expect(fieldCopyForSchemaKey(ru.settings.fieldLabels, 'display.show_reasoning')).toBe('Блоки рассуждений')
    expect(fieldCopyForSchemaKey(ru.settings.fieldDescriptions, 'display.show_reasoning')).toBe(
      'Показывать содержимое рассуждений, когда его предоставляет бэкенд.'
    )
  })

  it('uses Russian forms for count labels that do not include surrounding copy', () => {
    expect(ru.agents.workers(1)).toBe('1 воркер')
    expect(ru.agents.workers(2)).toBe('2 воркера')
    expect(ru.agents.tokens(5)).toBe('5 токенов')
    expect(ru.commandCenter.installTheme.installs('1')).toBe('1 установка')
    expect(ru.commandCenter.actions('2')).toBe('2 действия')
    expect(ru.preview.web.filesChanged(1, 'http://localhost:3000')).toBe(
      'Изменён 1 файл, обновление: http://localhost:3000'
    )
    expect(ru.desktop.skillCommandsAvailable(2)).toBe('Доступны 2 команды навыков.')
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
})
