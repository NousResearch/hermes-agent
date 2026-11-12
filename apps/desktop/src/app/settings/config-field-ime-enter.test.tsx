/**
 * A list-typed config field commits its draft on Enter by blurring the input,
 * so the Enter that *confirms an in-flight CJK composition* must not commit:
 * the draft would be parsed mid-word and written back to config. The field has
 * to go through the shared IME policy (`@/lib/ime`'s `isSubmitEnter`) instead
 * of a bare `event.key === 'Enter'` — the legacy `keyCode === 229` commit
 * Enter some IMEs (macOS Chinese) still fire after `compositionend` keeps
 * `isComposing` false, so the policy's second signal is what catches it.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ConfigField } from './config-field'

afterEach(cleanup)

function mount() {
  const onChange = vi.fn()

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ConfigField onChange={onChange} schema={{ type: 'list' }} schemaKey="tools.allowed" value={['111']} />
    </I18nProvider>
  )

  const input = screen.getByRole('textbox')
  // Enter commits by blurring the field, and jsdom only fires blur on the document's active element.
  input.focus()

  return { input, onChange }
}

it('commits the draft on a real Enter', () => {
  const { input, onChange } = mount()

  fireEvent.change(input, { target: { value: '111, 222' } })
  fireEvent.keyDown(input, { key: 'Enter' })

  expect(onChange).toHaveBeenCalledWith(['111', '222'])
})

it('leaves an in-flight IME composition Enter to the IME', () => {
  const { input, onChange } = mount()

  fireEvent.change(input, { target: { value: '中文' } })
  fireEvent.keyDown(input, { isComposing: true, key: 'Enter' })

  expect(onChange).not.toHaveBeenCalled()
})

it('leaves the legacy IME commit Enter (keyCode 229) to the IME', () => {
  const { input, onChange } = mount()

  fireEvent.change(input, { target: { value: '中文' } })
  fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })

  expect(onChange).not.toHaveBeenCalled()
})
