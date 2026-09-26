import type { ModelOptionProvider } from '@hermes/shared'
import { beforeEach, describe, expect, it } from 'vitest'

import {
  $customModels,
  addCustomModel,
  customModelCandidate,
  resetModelVisibilityKeepingCustoms,
  withCustomModels
} from './custom-models'
import {
  $knownModels,
  $visibleModels,
  defaultVisibleKeys,
  effectiveVisibleKeys,
  modelVisibilityKey,
  seedKnownModels
} from './model-visibility'

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

describe('resetModelVisibilityKeepingCustoms', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $customModels.set([])
    $visibleModels.set(null)
    $knownModels.set(null)
  })

  // An aggregator row: defaults are its featured shortlist, which a typed id
  // (appended after the catalog) is never part of.
  const openrouter = (): ModelOptionProvider => ({
    ...provider('openrouter', ['openai/gpt-5', 'openai/gpt-6', 'anthropic/claude-x']),
    featured_models: ['openai/gpt-6', 'anthropic/claude-x']
  })

  const qwen = provider('qwen', ['qwen3-coder'])

  it('clears the stale snapshot but keeps a custom model stored and shown', () => {
    addCustomModel('openrouter', 'acme/model-x', openrouter())
    // A pre-snapshot allowlist that left gpt-6 off, adopted as judged.
    $visibleModels.set(new Set([modelVisibilityKey('openrouter', 'acme/model-x')]))
    $knownModels.set(null)

    const catalog = withCustomModels([openrouter(), qwen], $customModels.get())
    seedKnownModels(catalog)
    expect(effectiveVisibleKeys($visibleModels.get(), catalog).has(modelVisibilityKey('openrouter', 'openai/gpt-6'))).toBe(
      false
    )

    resetModelVisibilityKeepingCustoms(catalog)

    const visible = effectiveVisibleKeys($visibleModels.get(), catalog)

    expect($customModels.get()).toEqual([{ model: 'acme/model-x', provider: 'openrouter' }])
    expect(window.localStorage.getItem('hermes.desktop.custom-models')).toContain('acme/model-x')
    expect(visible.has(modelVisibilityKey('openrouter', 'acme/model-x'))).toBe(true)

    for (const key of defaultVisibleKeys(catalog)) {
      expect(visible.has(key)).toBe(true)
    }

    // A provider with no custom model is not re-curated: it tracks live defaults.
    expect([...($visibleModels.get() ?? [])].some(key => key.startsWith('qwen::'))).toBe(false)
  })

  it('returns to the uncustomized state when every custom model is already a default', () => {
    addCustomModel('qwen', 'qwen-next', qwen)
    const catalog = withCustomModels([openrouter(), qwen], $customModels.get())

    resetModelVisibilityKeepingCustoms(catalog)

    expect($visibleModels.get()).toBeNull()
    expect($knownModels.get()).toBeNull()
    expect(effectiveVisibleKeys(null, catalog).has(modelVisibilityKey('qwen', 'qwen-next'))).toBe(true)
  })
})
