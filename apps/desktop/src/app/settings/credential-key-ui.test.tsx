import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { EnvVarInfo } from '@/types/hermes'

import { KeyField, type KeyRowProps } from './credential-key-ui'

afterEach(cleanup)

function makeRowProps(): KeyRowProps {
  return {
    edits: {},
    onClear: vi.fn(),
    onReveal: vi.fn(),
    onSave: vi.fn(),
    revealed: {},
    saving: null,
    setEdits: vi.fn()
  }
}

function renderKeyField(info: EnvVarInfo) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <KeyField info={info} rowProps={makeRowProps()} varKey="ANTHROPIC_API_KEY" />
    </I18nProvider>
  )
}

describe('KeyField', () => {
  it('renders a locked badge with no controls when managed_by is onepassword', () => {
    renderKeyField({
      advanced: false,
      category: 'provider',
      description: '',
      is_password: true,
      is_set: true,
      managed_by: 'onepassword',
      redacted_value: 'sk-...abcd',
      tools: [],
      url: null
    })

    expect(screen.getByText('sk-...abcd')).toBeInTheDocument()
    expect(screen.getByText('Managed via 1Password')).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('renders the normal editable field when managed_by is absent', () => {
    renderKeyField({
      advanced: false,
      category: 'provider',
      description: '',
      is_password: true,
      is_set: false,
      redacted_value: null,
      tools: [],
      url: null
    })

    expect(screen.getByPlaceholderText('Paste key')).toBeInTheDocument()
  })
})
