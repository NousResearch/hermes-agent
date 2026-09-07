import { describe, expect, it } from 'vitest'

import { isProviderSetupErrorMessage } from './provider-setup-errors'

describe('isProviderSetupErrorMessage', () => {
  it('matches generic missing-provider copy', () => {
    expect(isProviderSetupErrorMessage('No inference provider configured. Run `hermes model` to choose one.')).toBe(
      true
    )
    expect(isProviderSetupErrorMessage('No inference provider is configured.')).toBe(true)
    expect(isProviderSetupErrorMessage('No Hermes provider is configured.')).toBe(true)
    expect(isProviderSetupErrorMessage('set an API key (OPENROUTER_API_KEY) in ~/.hermes/.env')).toBe(true)
  })

  it('matches the exact empty-key warning emitted in session.info', () => {
    expect(
      isProviderSetupErrorMessage("No API key configured for provider 'openrouter'. First message will fail.")
    ).toBe(true)
  })

  it('does not match bare env var mentions from auxiliary warnings', () => {
    expect(isProviderSetupErrorMessage('OPENROUTER_API_KEY not set')).toBe(false)
    expect(isProviderSetupErrorMessage('Run `hermes setup` or set OPENROUTER_API_KEY.')).toBe(false)
    expect(
      isProviderSetupErrorMessage(
        '⚠ No auxiliary LLM provider configured — context compression will drop middle turns without a summary. Run `hermes setup` or set OPENROUTER_API_KEY.'
      )
    ).toBe(false)
    expect(isProviderSetupErrorMessage('OPENAI_API_KEY missing')).toBe(false)
    expect(isProviderSetupErrorMessage('ANTHROPIC_API_KEY not found')).toBe(false)
  })

  it('does not match non-provider runtime failures', () => {
    expect(
      isProviderSetupErrorMessage('Selected runtime is not available. setup.status reports configured credentials.')
    ).toBe(false)
  })

  it('returns false for empty input', () => {
    expect(isProviderSetupErrorMessage('')).toBe(false)
    expect(isProviderSetupErrorMessage(null)).toBe(false)
    expect(isProviderSetupErrorMessage(undefined)).toBe(false)
  })
})
