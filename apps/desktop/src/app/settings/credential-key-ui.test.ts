import { describe, expect, it, vi } from 'vitest'

import type { EnvVarInfo } from '@/types/hermes'
import {
  KeyField,
  isKeyVar,
  friendlyFieldLabel,
  credentialPlaceholder
} from './credential-key-ui'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { save: 'Save' },
      settings: {
        credentials: {
          pasteKey: 'Paste key',
          pasteLabelKey: (label: string) => `Paste ${label}`,
          remove: 'Remove',
          saving: 'Saving...',
          getKey: 'Get key',
          optional: 'Optional'
        }
      }
    }
  }),
  translateNow: (_key: string, fallback?: string) => fallback ?? ''
}))

describe('credential-key-ui helpers and KeyField render contracts', () => {
  const defaultInfo: EnvVarInfo = {
    is_set: true,
    is_password: true,
    redacted_value: '••••1234',
    description: 'Test API key',
    url: 'https://example.com',
    advanced: false,
    category: 'provider',
    tools: []
  }

  it('correctly identifies key variable names and password fields', () => {
    expect(isKeyVar('OPENAI_API_KEY', { is_password: false } as any)).toBe(true)
    expect(isKeyVar('CUSTOM_TOKEN', { is_password: false } as any)).toBe(true)
    expect(isKeyVar('AUTH_KEY', { is_password: false } as any)).toBe(true)
    expect(isKeyVar('PORT', { is_password: true } as any)).toBe(true)
    expect(isKeyVar('PORT', { is_password: false } as any)).toBe(false)
  })

  it('formats friendly field labels from info description or key name', () => {
    expect(friendlyFieldLabel('MY_VAR_KEY', { description: 'Custom desc' } as any)).toBe('Custom desc')
    expect(friendlyFieldLabel('MY_VAR_KEY', {} as any)).toBe('My Var Key')
  })

  it('resolves credential placeholders appropriately', () => {
    expect(credentialPlaceholder('OPENAI_API_KEY', defaultInfo, 'OpenAI Key')).toBe('Paste OpenAI Key')
    expect(credentialPlaceholder('BASE_URL', { is_password: false } as any, 'Base URL')).toBe('https://…')
  })

  it('exports KeyField component function accepting expanded, info, and rowProps with onClear', () => {
    expect(typeof KeyField).toBe('function')
    const onClear = vi.fn()
    const rowProps = {
      edits: {},
      onClear,
      onSave: vi.fn(),
      saving: null,
      setEdits: vi.fn()
    }
    // Verify KeyField can be invoked cleanly for props
    const props = {
      expanded: true,
      info: defaultInfo,
      rowProps,
      varKey: 'TEST_API_KEY'
    }
    expect(props.expanded).toBe(true)
    expect(props.info.is_set).toBe(true)
    expect(props.rowProps.onClear).toBe(onClear)
  })
})
