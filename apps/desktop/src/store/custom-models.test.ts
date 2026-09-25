import type { ModelOptionProvider } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { customModelCandidate, withCustomModels } from './custom-models'

const provider = (slug: string, models: string[]): ModelOptionProvider => ({
  models,
  name: slug,
  slug
})

describe('custom models', () => {
  it('offers a typed id only while no provider lists it', () => {
    const providers = [provider('openrouter', ['openai/gpt-5'])]

    expect(customModelCandidate('acme/model-x', providers)).toBe('acme/model-x')
    expect(customModelCandidate('OpenAI/GPT-5', providers)).toBeNull()
    expect(customModelCandidate('two words', providers)).toBeNull()
  })

  it('appends each remembered id under its own provider and keeps the input when nothing applies', () => {
    const providers = [provider('openrouter', ['openai/gpt-5']), provider('nous', ['hermes-4'])]

    const customs = [
      { model: 'acme/model-x', provider: 'openrouter' },
      { model: 'hermes-4', provider: 'nous' },
      { model: 'ghost', provider: 'missing' }
    ]

    const merged = withCustomModels(providers, customs)

    expect(merged.map(row => row.models)).toEqual([['openai/gpt-5', 'acme/model-x'], ['hermes-4']])
    expect(merged[1]).toBe(providers[1])
    expect(withCustomModels(providers, [])).toBe(providers)
  })
})
