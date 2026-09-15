import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'

import { I18nProvider } from '@/i18n'
import { ko } from '@/i18n/ko'

import { ConfigField } from './config-field'

afterEach(cleanup)

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
