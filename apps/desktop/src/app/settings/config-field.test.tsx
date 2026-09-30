import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'

import { I18nProvider } from '@/i18n'
import { ko } from '@/i18n/ko'
import type { ConfigFieldSchema } from '@/types/hermes'

import { ConfigField } from './config-field'

afterEach(cleanup)

test.each([
  { type: 'string', role: 'textbox', value: 'draft' },
  { type: 'number', role: 'spinbutton', value: 60 },
  { type: 'boolean', role: 'switch', value: true },
  { type: 'text', role: 'textbox', value: 'long description' },
  { type: 'list', role: 'textbox', value: ['one', 'two'] },
  { type: 'string', role: 'textbox', value: { key: 'value' } },
  { type: 'select', role: 'combobox', value: 'one', options: ['one', 'two'] },
  { type: 'select', role: 'combobox', value: 'one', options: ['one', 'two'], searchable: true }
])('names and describes a $type control for assistive technology', ({ role, value, ...schema }) => {
  render(
    <I18nProvider configClient={null} initialLocale="ko">
      <ConfigField
        onChange={() => {}}
        schema={schema as ConfigFieldSchema}
        schemaKey="terminal.docker_image"
        value={value}
      />
    </I18nProvider>
  )
  expect(
    screen.getByRole(role, {
      name: ko.settings.fieldLabels['terminal.dockerImage'],
      description: ko.settings.fieldDescriptions['terminal.dockerImage']
    })
  ).toBeTruthy()
})

test('labels free-input suggestions separately for repeated settings', () => {
  render(
    <I18nProvider configClient={null} initialLocale="ko">
      {['voice-a', 'voice-b'].map(value => (
        <ConfigField
          enumOptions={['alloy', 'nova']}
          key={value}
          onChange={() => {}}
          schema={{ type: 'string' }}
          schemaKey="tts.openai.voice"
          value={value}
        />
      ))}
    </I18nProvider>
  )
  const fields = screen.getAllByRole('textbox', { name: ko.settings.fieldLabels['tts.openai.voice'] })
  expect(fields).toHaveLength(2)
  expect(fields[0].getAttribute('aria-labelledby')).not.toBe(fields[1].getAttribute('aria-labelledby'))
})

test('renders distinct Korean field descriptions, including copy sharing a Latin product name', () => {
  render(
    <I18nProvider configClient={null} initialLocale="ko">
      <ConfigField onChange={() => {}} schema={{ type: 'number' }} schemaKey="approvals.timeout" value={60} />
      <ConfigField onChange={() => {}} schema={{ type: 'string' }} schemaKey="terminal.docker_image" value="" />
    </I18nProvider>
  )

  expect(screen.getByText(ko.settings.fieldDescriptions['approvals.timeout'])).toBeTruthy()
  expect(screen.getByText(ko.settings.fieldDescriptions['terminal.dockerImage'])).toBeTruthy()
})

test('still omits a schema description that only repeats its Korean label with punctuation', () => {
  const label = ko.settings.fieldLabels['agent.apiMaxRetries']
  const description = `${label}!`

  render(
    <I18nProvider configClient={null} initialLocale="ko">
      <ConfigField
        onChange={() => {}}
        schema={{ description, type: 'number' }}
        schemaKey="agent.api_max_retries"
        value={3}
      />
    </I18nProvider>
  )

  expect(screen.getByText(label)).toBeTruthy()
  expect(screen.queryByText(description)).toBeNull()
})
