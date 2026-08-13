import { describe, it, expect } from 'vitest'
import { generateProviderId, normalizeProviderName } from './custom-provider-config'

describe('normalizeProviderName', () => {
  it('lowercases and replaces spaces with dashes', () => {
    expect(normalizeProviderName('My Cool Provider')).toBe('my-cool-provider')
  })

  it('trims surrounding whitespace', () => {
    expect(normalizeProviderName('  Open AI ')).toBe('open-ai')
  })
})

describe('generateProviderId', () => {
  it('normalizes a simple name into a clean id', () => {
    expect(generateProviderId('llama', [])).toBe('llama')
  })

  it('lowercases and replaces spaces with dashes', () => {
    expect(generateProviderId('My Llama', [])).toBe('my-llama')
  })

  it('appends a numeric suffix on collision', () => {
    expect(generateProviderId('llama', ['llama'])).toBe('llama-2')
  })

  it('finds the next free suffix when several collide', () => {
    expect(generateProviderId('llama', ['llama', 'llama-2'])).toBe('llama-3')
  })

  it('falls back to "provider" when the name normalizes to nothing', () => {
    expect(generateProviderId('   ', [])).toBe('provider')
  })

  it('strips special characters', () => {
    expect(generateProviderId('Llama@#!', [])).toBe('llama')
  })

  it('detects collisions after normalization', () => {
    expect(generateProviderId('My Llama', ['my-llama'])).toBe('my-llama-2')
  })
})
