// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { AllowlistField } from './allowlist-field'

afterEach(cleanup)

it('edits allowed IDs using Korean controls while preserving the stored ID list', () => {
  const onEdit = vi.fn()
  render(
    <I18nProvider configClient={null} initialLocale="ko">
      <AllowlistField
        field={{
          advanced: false,
          description: '',
          is_list: true,
          is_password: false,
          is_set: true,
          key: 'SLACK_ALLOWED_USERS',
          prompt: '',
          redacted_value: null,
          required: false,
          url: null,
          value: '111,222'
        }}
        fieldId="slack-allowed"
        label="허용할 Slack 사용자 ID"
        onEdit={onEdit}
        pending={undefined}
        tools={null}
      />
    </I18nProvider>
  )

  fireEvent.click(screen.getAllByRole('button', { name: '항목 삭제' })[0])
  expect(onEdit).toHaveBeenLastCalledWith('SLACK_ALLOWED_USERS', '222')
  fireEvent.click(screen.getByRole('button', { name: 'ID 추가' }))
  const added = screen.getByRole('textbox', { name: '허용할 Slack 사용자 ID 2' })
  expect(added.getAttribute('placeholder')).toBe('ID를 입력하세요')
  fireEvent.change(added, { target: { value: '333, 444' } })
  expect(screen.getAllByRole('textbox').map(el => (el as HTMLInputElement).value)).toEqual(['222', '333', '444'])
  expect(onEdit).toHaveBeenLastCalledWith('SLACK_ALLOWED_USERS', '222,333,444')
})
