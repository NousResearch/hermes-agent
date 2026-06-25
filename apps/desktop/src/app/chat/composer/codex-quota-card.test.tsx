import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  useQuery: vi.fn()
}))

vi.mock('@tanstack/react-query', () => ({
  useQuery: (options: unknown) => mocks.useQuery(options)
}))

vi.mock('@/hermes', () => ({
  getCodexUsage: vi.fn()
}))

import { CodexQuotaCard } from './codex-quota-card'

describe('CodexQuotaCard', () => {
  beforeEach(() => {
    mocks.useQuery.mockReturnValue({ data: null, isPending: true })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('keys quota fetches by profile and provider so different Codex auth pools never share cache', () => {
    render(<CodexQuotaCard enabled profile="acewill-dev" provider="openai-codex" />)

    expect(mocks.useQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        enabled: true,
        queryKey: ['codex-usage', 'acewill-dev', 'openai-codex']
      })
    )
  })

  it('does not fetch Codex quota for non-Codex providers', () => {
    render(<CodexQuotaCard enabled profile="default" provider="anthropic" />)

    expect(mocks.useQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        enabled: false,
        queryKey: ['codex-usage', 'default', 'anthropic']
      })
    )
  })

  it('renders the full active profile and Codex account identity', () => {
    const fullAccountId = 'acct_user_01JYQ4ZLONGFULLACCOUNTIDENTIFIER_abcdef1234567890'

    mocks.useQuery.mockReturnValue({
      isPending: false,
      data: {
        account_id: fullAccountId,
        account_email: 'coder@example.com',
        account_label: 'work-codex-account',
        available: true,
        details: [],
        error: null,
        fetched_at: '2026-06-25T12:00:00Z',
        plan: 'Prolite',
        provider: 'openai-codex',
        source: 'usage_api',
        title: 'OpenAI Codex quota',
        windows: []
      }
    })

    render(<CodexQuotaCard enabled profile="acewill-dev" provider="openai-codex" />)

    expect(screen.getByText('Profile')).toBeTruthy()
    expect(screen.getByText('acewill-dev')).toBeTruthy()
    expect(screen.getByText('Account')).toBeTruthy()
    expect(
      screen.getByText(`coder@example.com (work-codex-account · ${fullAccountId})`)
    ).toBeTruthy()
  })
})
