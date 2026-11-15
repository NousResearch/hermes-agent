/**
 * The allowlist row's Enter-to-add must go through the shared IME-aware
 * predicate (`isSubmitEnter`), not a bare `event.key === 'Enter'` — a CJK
 * user's composition-commit Enter would commit the half-composed row and
 * clear the input before the candidate word is assembled. Same bug class as
 * the SectionNameDialog guard (see user-sections-ui.test.tsx).
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { MessagingEnvVarInfo } from '@/types/hermes'

import { AllowlistField } from './allowlist-field'

afterEach(cleanup)

const field: MessagingEnvVarInfo = {
  advanced: false,
  description: '',
  is_password: false,
  is_set: false,
  key: 'ALLOWED_USERS',
  prompt: '',
  redacted_value: null,
  required: false,
  url: null,
  value: ''
}

function mount() {
  const onEdit = vi.fn()
  render(
    <I18nProvider configClient={null}>
      <AllowlistField field={field} fieldId="field" label="Allowed user" onEdit={onEdit} pending={undefined} tools={null} />
    </I18nProvider>
  )

  return { onEdit }
}

describe('AllowlistField Enter-to-add', () => {
  it('adds a row and commits on a bare Enter', () => {
    mount()

    fireEvent.change(screen.getByLabelText('Allowed user 1'), { target: { value: 'alice' } })
    fireEvent.keyDown(screen.getByLabelText('Allowed user 1'), { key: 'Enter' })

    // The committed row keeps its entry and a fresh empty row takes focus.
    expect(screen.getByLabelText('Allowed user 1')).toHaveProperty('value', 'alice')
    expect(screen.getByLabelText('Allowed user 2')).toBeTruthy()
  })

  it('does not add a row on an in-flight IME composition Enter', () => {
    mount()

    fireEvent.change(screen.getByLabelText('Allowed user 1'), { target: { value: '中文' } })
    fireEvent.keyDown(screen.getByLabelText('Allowed user 1'), { isComposing: true, key: 'Enter' })

    expect(screen.queryByLabelText('Allowed user 2')).toBeNull()
  })

  it('does not add a row on the legacy IME commit Enter (keyCode 229)', () => {
    mount()

    fireEvent.change(screen.getByLabelText('Allowed user 1'), { target: { value: '中文' } })
    fireEvent.keyDown(screen.getByLabelText('Allowed user 1'), { key: 'Enter', keyCode: 229 })

    expect(screen.queryByLabelText('Allowed user 2')).toBeNull()
  })
})
