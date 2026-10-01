import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { FIELD_DESCRIPTIONS, FIELD_LABELS } from '@/app/settings/constants'
import { fieldCopyForSchemaKey } from '@/app/settings/field-copy'

import { TRANSLATIONS } from './catalog'
import { normalizeLocale } from './languages'
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

    expect(translateNow('boot.ready')).toBe(zh.boot.ready)
    expect(translateNow('assistant.tool.statusRecovered')).toBe(zh.assistant.tool.statusRecovered)
    expect(translateNow('boot.ready')).not.toBe(TRANSLATIONS.en.boot.ready)
  })

  it('passes arguments to function translations', () => {
    expect(translateNow('notifications.updateReadyMessage', 2)).toBe(
      TRANSLATIONS.en.notifications.updateReadyMessage(2)
    )
    expect(translateNow('notifications.updateReadyMessage', 2)).toContain('2')
  })

  it('keeps translated settings field copy addressable from schema keys', () => {
    const field = ['display', 'show_reasoning'].join('.')

    expect(fieldCopyForSchemaKey(zh.settings.fieldLabels, field)).toBeTypeOf('string')
    expect(fieldCopyForSchemaKey(zh.settings.fieldDescriptions, field)).toBeTypeOf('string')
  })

  it('falls back to English catalog/hybrid compression copy for untranslated locales', () => {
    const englishLabel = fieldCopyForSchemaKey(FIELD_LABELS, 'compression.mode')
    const englishDescription = fieldCopyForSchemaKey(FIELD_DESCRIPTIONS, 'compression.mode')

    expect(englishLabel).toBe('Compression Mode')
    expect(englishDescription).toMatch(/catalog/i)
    expect(englishDescription).toMatch(/hybrid/i)

    for (const locale of ['ar', 'ja', 'zh', 'zh-hant'] as const) {
      const labels = TRANSLATIONS[locale].settings.fieldLabels
      const descriptions = TRANSLATIONS[locale].settings.fieldDescriptions
      const label = fieldCopyForSchemaKey(labels, 'compression.mode')
      const description = fieldCopyForSchemaKey(descriptions, 'compression.mode')

      expect(label).toBeTruthy()
      expect(label).not.toBe('compression.mode')
      expect(label).not.toBe('settings.fieldLabels.compression.mode')
      expect(description).toBeTruthy()
      expect(description).not.toBe('compression.mode')
      expect(description).toMatch(/catalog|カタログ|目录|目錄|فهرس/i)
    }

    for (const unsupported of ['de', 'fr', 'xx-fake']) {
      const resolved = normalizeLocale(unsupported)
      expect(resolved).toBe('en')
      const copy = TRANSLATIONS[resolved].settings
      expect(fieldCopyForSchemaKey(copy.fieldLabels, 'compression.mode')).toBe(englishLabel)
      expect(fieldCopyForSchemaKey(copy.fieldDescriptions, 'compression.mode')).toBe(englishDescription)
    }

    const jaLabels = TRANSLATIONS.ja.settings.fieldLabels as Record<string, string | undefined>
    const original = jaLabels['compression.mode']
    try {
      delete jaLabels['compression.mode']
      expect(fieldCopyForSchemaKey(jaLabels as Record<string, string>, 'compression.mode')).toBeUndefined()
      expect(fieldCopyForSchemaKey(FIELD_LABELS, 'compression.mode')).toBe(englishLabel)
    } finally {
      if (original === undefined) {
        delete jaLabels['compression.mode']
      } else {
        jaLabels['compression.mode'] = original
      }
    }
  })

  it('falls back to English when the active locale cannot resolve a key', () => {
    const boot = TRANSLATIONS.ja.boot as { ready?: string }
    const originalReady = boot.ready

    try {
      boot.ready = undefined
      setRuntimeI18nLocale('ja')

      expect(translateNow('boot.ready')).toBe(TRANSLATIONS.en.boot.ready)
    } finally {
      boot.ready = originalReady
    }
  })

  it('returns the key when no locale can resolve a path', () => {
    setRuntimeI18nLocale('zh')

    expect(translateNow('missing.path')).toBe('missing.path')
  })
})
