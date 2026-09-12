import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { TRANSLATIONS } from '@/i18n/catalog'
import type { Locale } from '@/i18n/types'

import { ComboboxInput } from './combobox-input'

afterEach(cleanup)

const SHARED_LABEL_GAPS = [
  'browser.useRealProfile',
  'stt.echoTranscripts',
  'tts.deepinfra.model',
  'tts.deepinfra.voice'
]

const SHARED_DESCRIPTION_GAPS = [
  'browser.useRealProfile',
  'terminal.dockerImage',
  'terminal.singularityImage',
  'terminal.modalImage',
  'terminal.daytonaImage',
  'tts.xai.voiceId',
  'tts.xai.language',
  'tts.xai.speed',
  'tts.xai.autoSpeechTags',
  'tts.xai.optimizeStreamingLatency',
  'tts.xai.sampleRate',
  'tts.xai.bitRate',
  'tts.neutts.device',
  'stt.echoTranscripts'
]

const ZH_HANT_ONLY_GAPS = ['voice.voiceChatMode', 'voice.gptLive.voice', 'voice.gptLive.instructions']

describe('Settings i18n', () => {
  it.each([
    ['en', 'Show options'],
    ['zh', '显示选项'],
    ['zh-hant', '顯示選項']
  ] satisfies [Locale, string][])('renders combobox affordances in %s', (locale, expectedLabel) => {
    render(
      <I18nProvider configClient={null} initialLocale={locale}>
        <ComboboxInput onChange={() => {}} options={[]} value="" />
      </I18nProvider>
    )

    expect(screen.getByRole('button', { name: expectedLabel })).toBeTruthy()
  })

  it('provides reported Chinese field copy without falling through to English', () => {
    const en = TRANSLATIONS.en.settings
    const cases = [
      { locale: 'zh' as const, labels: SHARED_LABEL_GAPS, descriptions: SHARED_DESCRIPTION_GAPS },
      {
        locale: 'zh-hant' as const,
        labels: [...SHARED_LABEL_GAPS, ...ZH_HANT_ONLY_GAPS],
        descriptions: [...SHARED_DESCRIPTION_GAPS, ...ZH_HANT_ONLY_GAPS]
      }
    ]

    for (const { locale, labels, descriptions } of cases) {
      const settings = TRANSLATIONS[locale].settings

      for (const key of labels) {
        expect(settings.fieldLabels[key], `${locale} field label ${key}`).not.toBe(en.fieldLabels[key])
      }

      for (const key of descriptions) {
        expect(settings.fieldDescriptions[key], `${locale} field description ${key}`).not.toBe(
          en.fieldDescriptions[key]
        )
      }
    }

    expect(TRANSLATIONS.ja.settings.config.showOptions).toBe(en.config.showOptions)
  })
})
