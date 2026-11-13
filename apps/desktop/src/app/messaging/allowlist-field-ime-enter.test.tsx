/**
 * The allowlist editor's Enter opens a fresh row and clears the one being
 * edited, so an Enter that confirms an in-flight CJK composition must be left
 * to the IME — otherwise a user typing a translated name loses the half-typed
 * entry. Both IME signals matter: `isComposing` on the standard path, and the
 * legacy `keyCode === 229` commit Enter macOS Chinese IMEs fire after
 * `compositionend`. The shared policy lives in `@/lib/ime` (`isSubmitEnter`).
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { MessagingEnvVarInfo } from '@/hermes'
import { I18nProvider } from '@/i18n'

import { AllowlistField } from './allowlist-field'

afterEach(cleanup)

const field: MessagingEnvVarInfo = {
  advanced: false,
  description: 'Allowed user IDs.',
  is_list: true,
  is_password: false,
  is_set: true,
  key: 'TELEGRAM_ALLOWED_USERS',
  prompt: 'Allowed users',
  redacted_value: null,
  required: false,
  url: null,
  value: '111'
}

function mount() {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <AllowlistField
        field={field}
        fieldId="allowlist"
        label="Allowed users"
        onEdit={vi.fn()}
        pending={undefined}
        tools={null}
      />
    </I18nProvider>
  )

  return { input: screen.getAllByRole('textbox')[0] }
}

const rowCount = () => screen.getAllByRole('textbox').length

it('opens a new row on a real Enter', () => {
  const { input } = mount()

  fireEvent.change(input, { target: { value: '222' } })
  fireEvent.keyDown(input, { key: 'Enter' })

  expect(rowCount()).toBe(2)
})

it('leaves an in-flight IME composition Enter to the IME', () => {
  const { input } = mount()

  fireEvent.change(input, { target: { value: '中文' } })
  fireEvent.keyDown(input, { isComposing: true, key: 'Enter' })

  expect(rowCount()).toBe(1)
})

it('leaves the legacy IME commit Enter (keyCode 229) to the IME', () => {
  const { input } = mount()

  fireEvent.change(input, { target: { value: '中文' } })
  fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })

  expect(rowCount()).toBe(1)
})
