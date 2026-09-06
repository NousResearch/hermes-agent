import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ProviderAvatar, providerBrand } from './provider-avatar'

afterEach(cleanup)

describe('providerBrand', () => {
  it('maps OpenAI Codex profiles to OpenAI', () => {
    expect(providerBrand('openai-codex', 'gpt-5.6-luna')?.label).toBe('OpenAI')
  })

  it('maps xAI OAuth profiles to the Grok mark', () => {
    expect(providerBrand('xai-oauth', 'grok-4.6')).toMatchObject({ id: 'xai', label: 'Grok' })
  })

  it('maps Anthropic profiles to Claude', () => {
    expect(providerBrand('anthropic', 'claude-fable-5-1')?.label).toBe('Claude')
  })

  it('uses the model as a fallback when the provider is custom', () => {
    expect(providerBrand('custom:gateway', 'anthropic/claude-sonnet-4.6')?.label).toBe('Claude')
  })

  it('returns no brand for an unknown provider and model', () => {
    expect(providerBrand('custom:gateway', 'local-model')).toBeNull()
  })
})

describe('ProviderAvatar', () => {
  it('renders a provider glyph with an accessible provider label', () => {
    render(<ProviderAvatar model="gpt-5.6-luna" name="Luna" provider="openai-codex" />)

    expect(screen.getByRole('img', { name: 'Luna · OpenAI' })).toBeTruthy()
    expect(screen.getByRole('img', { name: 'Luna · OpenAI' }).querySelector('svg')).not.toBeNull()
  })
})
