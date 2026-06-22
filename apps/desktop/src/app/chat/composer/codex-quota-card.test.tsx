import { cleanup, render } from '@testing-library/react'
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
})
