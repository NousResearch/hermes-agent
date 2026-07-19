import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { fieldCopyForSchemaKey } from '@/app/settings/field-copy'

import { TRANSLATIONS } from './catalog'
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

  it('translates the core desktop surface in Korean', () => {
    setRuntimeI18nLocale('ko')

    expect(translateNow('language.switchTo')).toBe('언어 전환')
    expect(translateNow('settings.appearance.title')).toBe('화면 설정')
    expect(translateNow('settings.sections.safety')).toBe('승인 및 실행')
    expect(translateNow('settings.model.auxiliaryTitle')).toBe('보조 모델')
    expect(translateNow('settings.model.moa.aggregator')).toBe('종합 모델')
    expect(translateNow('common.tryHint', 'discord')).toBe('다음 검색어로 시도해 보세요: “discord”')
    expect(translateNow('common.tryHint', '검색')).toBe('다음 검색어로 시도해 보세요: “검색”')
    expect(translateNow('messaging.fieldCopy.TELEGRAM_ALLOWED_USERS.placeholder')).toBe(
      'Telegram 사용자 ID를 쉼표로 구분'
    )
    expect(translateNow('messaging.fieldCopy.TELEGRAM_ALLOW_ALL_USERS.placeholder')).toBe('true 또는 false')
    expect(translateNow('sidebar.nav.new-session')).toBe('새 세션')
    expect(translateNow('sidebar.nav.messaging')).toBe('메시지 연동')
    expect(translateNow('sidebar.nav.artifacts')).toBe('작업 결과')
    expect(translateNow('messaging.search')).toBe('메시지 연동 검색…')
    expect(translateNow('messaging.fieldCopy.TELEGRAM_HOME_CHANNEL.label')).toBe('기본 채팅 ID')
    expect(translateNow('messaging.platformDescription.telegram')).toBe(
      'Telegram 개인 메시지, 그룹, 주제에서 Hermes를 사용합니다.'
    )
    expect(translateNow('messaging.platformEnabled', 'Telegram')).toBe('Telegram 연동을 켰습니다')
    expect(translateNow('messaging.platformDisabled', 'Telegram')).toBe('Telegram 연동을 껐습니다')
    expect(translateNow('artifacts.noArtifactsTitle')).toBe('작업 결과가 없습니다')
    expect(translateNow('composer.placeholderFollowUp')).toBe('후속 메시지 보내기')
  })

  it('keeps dynamic Latin values free of invalid Korean particles', () => {
    setRuntimeI18nLocale('ko')

    expect(translateNow('fileMenu.deleteTitle', 'README.md')).toBe('“README.md” 항목을 삭제할까요?')
    expect(translateNow('settings.appearance.pet.noMatch', 'Slack')).toBe(
      '“Slack” 검색 결과에 맞는 마스코트가 없습니다.'
    )
    expect(translateNow('messaging.keyCleared', 'TELEGRAM_BOT_TOKEN')).toBe('TELEGRAM_BOT_TOKEN 값을 지웠습니다')
    expect(translateNow('messaging.failedUpdate', 'Telegram')).toBe('Telegram 업데이트에 실패했습니다')
    expect(translateNow('messaging.failedSave', 'Telegram')).toBe('Telegram 저장에 실패했습니다')
    expect(translateNow('messaging.failedClear', 'TELEGRAM_BOT_TOKEN')).toBe(
      'TELEGRAM_BOT_TOKEN 값을 지우지 못했습니다'
    )
    expect(translateNow('sidebar.noMatch', 'Slack')).toBe('“Slack” 검색 결과가 없습니다.')
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
