import { describe, expect, it } from 'vitest'

import { isProviderSetupErrorCode, isProviderSetupErrorMessage } from './provider-setup-errors'

describe('isProviderSetupErrorMessage', () => {
  it('matches generic missing-provider copy', () => {
    expect(isProviderSetupErrorMessage('No inference provider configured. Run `hermes model` to choose one.')).toBe(
      true
    )
    expect(isProviderSetupErrorMessage('No inference provider is configured.')).toBe(true)
    expect(isProviderSetupErrorMessage('No Hermes provider is configured.')).toBe(true)
    expect(isProviderSetupErrorMessage('set an API key (OPENROUTER_API_KEY) in ~/.hermes/.env')).toBe(true)
  })

  it('matches what agent init actually says on a blank install', () => {
    // tui_gateway/user_messages.py::agent_init_failed_message wrapping agent/agent_init.py's raise.
    expect(
      isProviderSetupErrorMessage(
        'Hermes could not start the assistant for this session. Details: No LLM provider configured. Run `hermes model` to select a provider, or run `hermes setup` for first-time configuration. Check the model and provider with /model, or run `hermes setup` in a terminal to reconfigure.'
      )
    ).toBe(true)
  })

  it('matches an explicit provider that has no credentials', () => {
    // agent/auxiliary_unavailable.py::missing_provider_credentials_message, both shapes.
    expect(
      isProviderSetupErrorMessage(
        "Provider 'openai' is set in config.yaml but no API key was found. Set the OPENAI_API_KEY environment variable, or switch to a different provider with `hermes model`."
      )
    ).toBe(true)
    expect(
      isProviderSetupErrorMessage(
        "Provider 'anthropic' is set in config.yaml but no credentials were found. Run `hermes auth add anthropic` to sign in, or switch to a different provider with `hermes model`."
      )
    ).toBe(true)
  })

  it('reads the gateway code in preference to any sentence', () => {
    expect(isProviderSetupErrorCode('provider_not_configured')).toBe(true)
    expect(isProviderSetupErrorCode('disk_full')).toBe(false)
    expect(isProviderSetupErrorCode(undefined)).toBe(false)
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
