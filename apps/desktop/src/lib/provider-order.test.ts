import { describe, expect, it } from 'vitest'

import { compareModelProviders, isProviderConnected, sortModelProviders } from './provider-order'

const openrouter = { name: 'OpenRouter', slug: 'openrouter', authenticated: true }
const copilot = { name: 'GitHub Copilot', slug: 'copilot', authenticated: true }
const xai = { name: 'xAI', slug: 'xai-oauth', authenticated: true }
const nous = { name: 'Nous Portal', slug: 'nous' } // flag omitted → connected
const broken = { name: 'Anthropic', slug: 'anthropic', authenticated: false }

describe('isProviderConnected', () => {
  it('treats missing authenticated as connected', () => {
    expect(isProviderConnected({ authenticated: undefined })).toBe(true)
  })

  it('treats true as connected and false as not', () => {
    expect(isProviderConnected({ authenticated: true })).toBe(true)
    expect(isProviderConnected({ authenticated: false })).toBe(false)
  })
})

describe('compareModelProviders / sortModelProviders', () => {
  it('sorts alphabetically when every provider is connected and none is current', () => {
    expect(sortModelProviders([xai, openrouter, copilot, nous]).map(p => p.slug)).toEqual([
      'copilot',
      'nous',
      'openrouter',
      'xai-oauth'
    ])
  })

  it('pins the current provider first without reshuffling peers', () => {
    const ordered = sortModelProviders([xai, openrouter, copilot, nous], 'xai-oauth')
    expect(ordered.map(p => p.slug)).toEqual(['xai-oauth', 'copilot', 'nous', 'openrouter'])
  })

  it('accepts the model-options payload provider field as the current slug', () => {
    // Edit Models / other catalog consumers get current via payload.provider
    // (inventory.build_models_payload), not a separate session store.
    const payload = {
      provider: 'xai-oauth',
      providers: [openrouter, xai, copilot]
    }
    const ordered = sortModelProviders(payload.providers, payload.provider)
    expect(ordered.map(p => p.slug)).toEqual(['xai-oauth', 'copilot', 'openrouter'])
  })

  it('keeps connected providers above needs-setup rows', () => {
    const ordered = sortModelProviders([broken, xai, openrouter], 'openrouter')
    expect(ordered.map(p => p.slug)).toEqual(['openrouter', 'xai-oauth', 'anthropic'])
  })

  it('matches current slug case-insensitively and by trim', () => {
    expect(compareModelProviders(xai, openrouter, ' XAI-OAUTH ')).toBeLessThan(0)
    expect(compareModelProviders(openrouter, xai, 'xai-oauth')).toBeGreaterThan(0)
  })

  it('does not move an already-current provider relative to itself', () => {
    expect(compareModelProviders(xai, xai, 'xai-oauth')).toBe(0)
  })
})
