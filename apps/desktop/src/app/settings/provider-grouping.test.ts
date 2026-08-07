import { describe, expect, it } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import { classifyProvider, flattenGroups, groupProviders, isConfigurableProvider } from './provider-grouping'

const p = (over: Partial<ModelOptionProvider> & Pick<ModelOptionProvider, 'slug' | 'name'>): ModelOptionProvider => ({
  models: [],
  ...over
})

describe('classifyProvider', () => {
  it('classifies slug "local" as local', () => {
    expect(classifyProvider(p({ slug: 'local', name: 'Local' }))).toBe('local')
  })

  it('classifies an authenticated provider as configured', () => {
    expect(classifyProvider(p({ slug: 'openai', name: 'OpenAI', authenticated: true }))).toBe('configured')
  })

  it('classifies a user-defined provider as configured', () => {
    expect(classifyProvider(p({ slug: 'custom:lab', name: 'Lab', is_user_defined: true }))).toBe('configured')
  })

  it('classifies an enabled provider with models as configured', () => {
    expect(classifyProvider(p({ slug: 'anthropic', name: 'Anthropic', models: ['claude'] }))).toBe('configured')
  })

  it('classifies an unauthenticated, model-less provider as unconfigured', () => {
    expect(classifyProvider(p({ slug: 'deepseek', name: 'DeepSeek', authenticated: false }))).toBe('unconfigured')
  })
})

describe('groupProviders', () => {
  it('orders groups Local → Configured → Unconfigured', () => {
    const groups = groupProviders([
      p({ slug: 'deepseek', name: 'DeepSeek', authenticated: false }),
      p({ slug: 'openai', name: 'OpenAI', authenticated: true }),
      p({ slug: 'local', name: 'Local' })
    ])

    expect(groups.map(g => g.id)).toEqual(['local', 'configured', 'unconfigured'])
  })

  it('omits empty groups', () => {
    const groups = groupProviders([p({ slug: 'openai', name: 'OpenAI', authenticated: true })])
    expect(groups.map(g => g.id)).toEqual(['configured'])
  })

  it('puts active providers before disabled within a group', () => {
    const groups = groupProviders([
      p({ slug: 'openai', name: 'OpenAI', authenticated: true, enabled: false }),
      p({ slug: 'anthropic', name: 'Anthropic', authenticated: true, enabled: true })
    ])

    expect(groups[0].providers.map(pr => pr.slug)).toEqual(['anthropic', 'openai'])
  })

  it('sorts alphabetically within the same active state', () => {
    const groups = groupProviders([
      p({ slug: 'zoo', name: 'Zoo', authenticated: true }),
      p({ slug: 'alpha', name: 'Alpha', authenticated: true })
    ])

    expect(groups[0].providers.map(pr => pr.slug)).toEqual(['alpha', 'zoo'])
  })

  it('keeps disabled providers grouped together regardless of name', () => {
    const groups = groupProviders([
      p({ slug: 'aaa', name: 'Aaa', authenticated: true, enabled: false }),
      p({ slug: 'bbb', name: 'Bbb', authenticated: true, enabled: true })
    ])

    // Bbb (active) must precede Aaa (disabled) even though "Aaa" < "Bbb".
    expect(groups[0].providers.map(pr => pr.slug)).toEqual(['bbb', 'aaa'])
  })
})

describe('flattenGroups', () => {
  it('returns providers in group-then-order sequence', () => {
    const groups = groupProviders([
      p({ slug: 'deepseek', name: 'DeepSeek', authenticated: false }),
      p({ slug: 'openai', name: 'OpenAI', authenticated: true }),
      p({ slug: 'local', name: 'Local' })
    ])

    expect(flattenGroups(groups).map(pr => pr.slug)).toEqual(['local', 'openai', 'deepseek'])
  })
})

describe('isConfigurableProvider', () => {
  it('is true for an unconfigured api_key provider with a key_env', () => {
    expect(
      isConfigurableProvider(
        p({ slug: 'openai', name: 'OpenAI', auth_type: 'api_key', key_env: 'OPENAI_API_KEY', authenticated: false })
      )
    ).toBe(true)
  })

  it('is false when the provider is already authenticated', () => {
    expect(
      isConfigurableProvider(
        p({ slug: 'openai', name: 'OpenAI', auth_type: 'api_key', key_env: 'OPENAI_API_KEY', authenticated: true })
      )
    ).toBe(false)
  })

  it('is false for a user-defined custom provider', () => {
    expect(
      isConfigurableProvider(
        p({ slug: 'custom:lab', name: 'Lab', auth_type: 'api_key', key_env: 'LAB_API_KEY', is_user_defined: true })
      )
    ).toBe(false)
  })

  it('is false for an oauth provider', () => {
    expect(
      isConfigurableProvider(p({ slug: 'google', name: 'Google', auth_type: 'oauth_google', key_env: 'GOOGLE_API_KEY' }))
    ).toBe(false)
  })

  it('is false when key_env is missing', () => {
    expect(isConfigurableProvider(p({ slug: 'openai', name: 'OpenAI', auth_type: 'api_key' }))).toBe(false)
  })

  it('is false when auth_type is undefined', () => {
    expect(isConfigurableProvider(p({ slug: 'openai', name: 'OpenAI', key_env: 'OPENAI_API_KEY' }))).toBe(false)
  })
})
