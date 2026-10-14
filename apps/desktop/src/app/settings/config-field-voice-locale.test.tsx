import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider, TRANSLATIONS } from '@/i18n'

import { ConfigField } from './config-field'
import { fieldCopyForSchemaKey } from './field-copy'

afterEach(cleanup)

it('renders the translated Echo Transcripts copy without changing its toggle value', () => {
  for (const locale of ['zh', 'zh-hant'] as const) {
    const onChange = vi.fn()
    const t = TRANSLATIONS[locale]

    const { unmount } = render(
      <I18nProvider configClient={null} initialLocale={locale}>
        <ConfigField onChange={onChange} schema={{ type: 'boolean' }} schemaKey="stt.echo_transcripts" value={false} />
      </I18nProvider>
    )

    expect(screen.getByText(fieldCopyForSchemaKey(t.settings.fieldLabels, 'stt.echo_transcripts')!)).toBeTruthy()
    expect(screen.getByText(fieldCopyForSchemaKey(t.settings.fieldDescriptions, 'stt.echo_transcripts')!)).toBeTruthy()
    expect(screen.queryByText('Echo Transcripts')).toBeNull()
    fireEvent.click(screen.getByRole('switch'))
    expect(onChange).toHaveBeenCalledWith(true)
    unmount()
  }
})

it('preserves distinct Unicode descriptions but suppresses label and schema-key repetitions', () => {
  const cases = [
    ['説明', '音声を表示します'],
    ['Голос', 'Показывать текст'],
    ['الصوت', 'عرض النص'],
    ['किताब', 'कताब'],
    ['café', 'cafe']
  ]

  for (const [label, description] of cases) {
    const schemaKey = `custom.${label}`

    const field = (copy: string) => (
      <ConfigField
        onChange={() => {}}
        schema={{ type: 'boolean', description: copy }}
        schemaKey={schemaKey}
        value={false}
      />
    )

    const { rerender, unmount } = render(field(description))
    expect(screen.getByText(description)).toBeTruthy()

    for (const duplicate of [`${label.toUpperCase()}!`, `${schemaKey}!`, `${label.normalize('NFD')}!`]) {
      rerender(field(duplicate))
      expect(screen.queryByText(duplicate)).toBeNull()
    }

    rerender(field(''))
    expect(screen.queryByText(description)).toBeNull()
    unmount()
  }

  const { unmount } = render(
    <ConfigField
      descriptionExtra={<span>Extra help</span>}
      onChange={() => {}}
      schema={{ type: 'boolean', description: 'CUSTOM setting!' }}
      schemaKey="custom_setting"
      value={false}
    />
  )

  expect(screen.queryByText('CUSTOM setting!')).toBeNull()
  expect(screen.getByText('Extra help')).toBeTruthy()
  unmount()
})
