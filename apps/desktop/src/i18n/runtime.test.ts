import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { fieldCopyForSchemaKey } from '@/app/settings/field-copy'
import { credentialRowLabel, providerGroupDescription } from '@/app/settings/credential-key-ui'

import { TRANSLATIONS } from './catalog'
import { setRuntimeI18nLocale, translateNow } from './runtime'
import { ar } from './ar'
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
    expect(translateNow('settings.sections.model')).toBe('النموذج')
    expect(translateNow('settings.nav.providers')).toBe('المزودون')
    expect(translateNow('settings.gateway.title')).toBe('اتصال البوابة')
    expect(translateNow('settings.mcp.emptyTitle')).toBe('لا توجد خوادم MCP')
    expect(translateNow('settings.sessions.archivedTitle')).toBe('الجلسات المؤرشفة')
    expect(translateNow('settings.uninstall.dangerZone')).toBe('منطقة الخطر')
    expect(translateNow('settings.appearance.translucencyTitle')).toBe('شفافية النافذة')
    expect(translateNow('onboarding.connected')).toBe('متصل')
  })

  it('uses Arabic copy for credential and provider rows', () => {
    setRuntimeI18nLocale('ar')

    expect(credentialRowLabel('OPENROUTER_API_KEY', { is_password: true } as never)).toBe('مفتاح OpenRouter API')
    expect(providerGroupDescription('OpenRouter', 'Aggregator for hundreds of frontier models')).toBe(
      'مجمّع يتيح مئات النماذج المتقدمة عبر مفتاح واحد.'
    )
  })

  it('keeps translated settings field copy addressable from schema keys', () => {
    const field = ['display', 'show_reasoning'].join('.')

    expect(fieldCopyForSchemaKey(zh.settings.fieldLabels, field)).toBe('推理过程块')
    expect(fieldCopyForSchemaKey(zh.settings.fieldDescriptions, field)).toBe('当后端提供推理内容时予以显示。')
  })

  it('keeps Arabic settings field copy addressable from schema keys', () => {
    expect(fieldCopyForSchemaKey(ar.settings.fieldLabels, 'terminal.cwd')).toBe('مجلد العمل')
    expect(fieldCopyForSchemaKey(ar.settings.fieldLabels, 'approvals.mode')).toBe('نمط الموافقات')
    expect(fieldCopyForSchemaKey(ar.settings.fieldLabels, 'memory.memory_enabled')).toBe('الذاكرة المستمرة')
    expect(fieldCopyForSchemaKey(ar.settings.fieldDescriptions, 'terminal.cwd')).toBe(
      'مجلد المشروع الافتراضي لعمل الأدوات والطرفية.'
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
