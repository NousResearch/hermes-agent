import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { EnvVarInfo } from '@/types/hermes'

import { CredentialKeyCard } from './credential-key-ui'

// MemoryConnect (rendered for connect-capable credentials) probes the backend
// on mount via these bridge calls; stub them so the button surfaces.
const getMemoryProviderOAuthStatus = vi.fn()
const startMemoryProviderOAuth = vi.fn()

vi.mock('@/hermes', () => ({
  getMemoryProviderOAuthStatus: (provider: string) => getMemoryProviderOAuthStatus(provider),
  startMemoryProviderOAuth: (provider: string) => startMemoryProviderOAuth(provider)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const rowProps = {
  edits: {},
  revealed: {},
  saving: null,
  setEdits: vi.fn(),
  onSave: vi.fn(),
  onClear: vi.fn(),
  onReveal: vi.fn()
}

function makeInfo(overrides: Partial<EnvVarInfo> = {}): EnvVarInfo {
  return {
    advanced: false,
    category: 'tool',
    description: 'Hindsight API key for graph-aware persistent memory',
    is_password: true,
    is_set: false,
    redacted_value: null,
    tools: [],
    url: 'https://hindsight.vectorize.io',
    ...overrides
  }
}

function renderCard(info: EnvVarInfo, varKey: string) {
  return render(
    <CredentialKeyCard
      expanded
      info={info}
      label="Hindsight API Key"
      onExpand={vi.fn()}
      onToggle={vi.fn()}
      placeholder="Enter key"
      rowProps={rowProps}
      varKey={varKey}
    />
  )
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('CredentialKeyCard — connect_provider', () => {
  it('renders the browser Connect button (not the external "Get a key" link) when connect_provider is set', async () => {
    getMemoryProviderOAuthStatus.mockResolvedValue({ state: 'idle', detail: '', connected: false, auth: null })

    renderCard(makeInfo({ connect_provider: 'hindsight' }), 'HINDSIGHT_API_KEY')

    // MemoryConnect probes on mount, then surfaces a Connect button.
    await screen.findByText('Connect')
    expect(getMemoryProviderOAuthStatus).toHaveBeenCalledWith('hindsight')
    // The plain external "Get a key" anchor is replaced by the connect flow.
    expect(screen.queryByRole('link')).toBeNull()
  })

  it('renders the external "Get a key" link when connect_provider is absent', () => {
    renderCard(makeInfo(), 'SOME_API_KEY')

    expect(getMemoryProviderOAuthStatus).not.toHaveBeenCalled()
    const link = screen.getByRole('link')
    expect(link.getAttribute('href')).toBe('https://hindsight.vectorize.io')
  })
})
