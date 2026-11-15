/**
 * ListField's Enter-to-commit must go through the shared IME-aware predicate
 * (`isSubmitEnter`), not a bare `event.key === 'Enter'` — a CJK user's
 * composition-commit Enter would commit the half-composed draft and blur the
 * field before the candidate word is assembled. Same bug class as the
 * SectionNameDialog guard (see user-sections-ui.test.tsx).
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ConfigField } from './config-field'

afterEach(cleanup)

function mount() {
  const onChange = vi.fn()
  render(
    <I18nProvider configClient={null}>
      <ConfigField onChange={onChange} schema={{ type: 'list' }} schemaKey="test.list" value={['one']} />
    </I18nProvider>
  )

  const input = screen.getByRole('textbox')
  input.focus()

  // ListField's change handler only stages the draft; onChange fires on commit.
  return { input, onChange }
}

describe('ListField Enter-to-commit', () => {
  it('commits the parsed draft on a bare Enter', () => {
    const { input, onChange } = mount()

    fireEvent.change(input, { target: { value: 'one, two' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    // Enter commits by blurring the field; jsdom moves focus synchronously but
    // needs the blur event dispatched for React's onBlur to run.
    expect(document.activeElement).not.toBe(input)
    fireEvent.blur(input)

    expect(onChange).toHaveBeenCalledWith(['one', 'two'])
  })

  it('does not commit on an in-flight IME composition Enter', () => {
    const { input, onChange } = mount()

    fireEvent.change(input, { target: { value: 'one, 中文' } })
    fireEvent.keyDown(input, { isComposing: true, key: 'Enter' })

    // The composition Enter must not blur the field (blurring is what commits
    // the half-composed draft); a deliberate blur still commits afterwards.
    expect(document.activeElement).toBe(input)
    fireEvent.blur(input)
    expect(onChange).toHaveBeenCalledWith(['one', '中文'])
  })

  it('does not commit on the legacy IME commit Enter (keyCode 229)', () => {
    const { input, onChange } = mount()

    fireEvent.change(input, { target: { value: 'one, 中文' } })
    fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })

    expect(document.activeElement).toBe(input)
    fireEvent.blur(input)
    expect(onChange).toHaveBeenCalledWith(['one', '中文'])
  })
})
