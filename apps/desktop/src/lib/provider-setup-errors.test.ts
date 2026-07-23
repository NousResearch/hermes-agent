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

  it('matches Codex OAuth setup errors', () => {
    expect(
      isProviderSetupErrorMessage(
        'No Codex credentials stored. Run `hermes auth add openai-codex` to authenticate OpenAI OAuth for this Hermes home/profile.'
      )
    ).toBe(true)
    expect(isProviderSetupErrorMessage('Codex auth is missing refresh_token. Run `hermes auth` to re-authenticate.')).toBe(true)
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
