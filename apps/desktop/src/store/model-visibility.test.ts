import { describe, expect, it } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import {
  collapseModelFamilies,
  compareModelFamilies,
  defaultVisibleKeys,
  effectiveVisibleKeys,
  emptyProviderSentinelKey,
  isProviderSentinel,
  modelVisibilityKey,
  resolveVisibleKeys,
  toggleModelVisibility
} from './model-visibility'

const provider = (slug: string, models: string[]): ModelOptionProvider => ({
  models,
  name: slug,
  slug
})

describe('model visibility', () => {
  it('keeps newly configured providers visible when stored choices are stale', () => {
    const stored = new Set([modelVisibilityKey('copilot', 'claude-sonnet-4.6')])

    const visible = effectiveVisibleKeys(stored, [
      provider('copilot', ['claude-sonnet-4.6']),
      provider('local-ollama', ['qwen3:latest', 'llama3.2:latest'])
    ])

    expect(visible.has(modelVisibilityKey('copilot', 'claude-sonnet-4.6'))).toBe(true)
    expect(visible.has(modelVisibilityKey('local-ollama', 'qwen3:latest'))).toBe(true)
    expect(visible.has(modelVisibilityKey('local-ollama', 'llama3.2:latest'))).toBe(true)
  })

  it('does not re-add models from a provider that already has stored choices', () => {
    const stored = new Set([modelVisibilityKey('local-ollama', 'qwen3:latest')])

    const visible = effectiveVisibleKeys(stored, [provider('local-ollama', ['qwen3:latest', 'llama3.2:latest'])])

    expect(visible.has(modelVisibilityKey('local-ollama', 'qwen3:latest'))).toBe(true)
    expect(visible.has(modelVisibilityKey('local-ollama', 'llama3.2:latest'))).toBe(false)
  })

  it('preserves hidden-provider sentinel without re-adding defaults', () => {
    // User explicitly hid all models for "nous" — sentinel marks this choice.
    const stored = new Set([emptyProviderSentinelKey('nous')])

    const visible = effectiveVisibleKeys(stored, [
      provider('nous', ['hermes-3-llama-3.1-70b', 'hermes-3-llama-3.1-8b']),
      provider('ollama', ['qwen3:latest'])
    ])

    expect(visible.has(modelVisibilityKey('nous', 'hermes-3-llama-3.1-70b'))).toBe(false)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-3-llama-3.1-8b'))).toBe(false)
    // Sentinel itself is stripped from the result.
    expect(visible.has(emptyProviderSentinelKey('nous'))).toBe(false)
    // Other providers still get defaults.
    expect(visible.has(modelVisibilityKey('ollama', 'qwen3:latest'))).toBe(true)
  })

  it('restores model when toggling on after hiding all', () => {
    // Simulates: user hid all "nous" models, then toggles one back on.
    const stored = new Set([emptyProviderSentinelKey('nous'), modelVisibilityKey('ollama', 'qwen3:latest')])

    // After toggle: sentinel removed, one model added.
    const afterToggle = new Set(stored)
    afterToggle.delete(emptyProviderSentinelKey('nous'))
    afterToggle.add(modelVisibilityKey('nous', 'hermes-3-llama-3.1-70b'))

    const visible = effectiveVisibleKeys(afterToggle, [
      provider('nous', ['hermes-3-llama-3.1-70b', 'hermes-3-llama-3.1-8b']),
      provider('ollama', ['qwen3:latest'])
    ])

    expect(visible.has(modelVisibilityKey('nous', 'hermes-3-llama-3.1-70b'))).toBe(true)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-3-llama-3.1-8b'))).toBe(false)
  })

  it('folds a date-pinned snapshot into its rolling alias when present', () => {
    const families = collapseModelFamilies(['claude-opus-4-5', 'claude-opus-4-5-20251101'])

    expect(families.map(f => f.id)).toEqual(['claude-opus-4-5'])
  })

  it('keeps a date-pinned snapshot standing alone when it has no alias', () => {
    const families = collapseModelFamilies(['claude-opus-4-5-20251101', 'claude-haiku-4-5-20251001'])

    expect(families.map(f => f.id)).toEqual(['claude-opus-4-5-20251101', 'claude-haiku-4-5-20251001'])
  })

  it('sentinel key helper produces correct format', () => {
    expect(emptyProviderSentinelKey('openai')).toBe('openai::')
    expect(isProviderSentinel('openai::')).toBe(true)
    expect(isProviderSentinel('openai::gpt-4o')).toBe(false)
  })

  it('resolveVisibleKeys preserves sentinels that effectiveVisibleKeys strips', () => {
    const stored = new Set([emptyProviderSentinelKey('nous')])
    const providers = [provider('nous', ['hermes-x', 'hermes-y']), provider('ollama', ['qwen3:latest'])]

    const resolved = resolveVisibleKeys(stored, providers)
    expect(resolved.has(emptyProviderSentinelKey('nous'))).toBe(true)
    expect(resolved.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(false)
    // Un-customized providers still expand to their defaults.
    expect(resolved.has(modelVisibilityKey('ollama', 'qwen3:latest'))).toBe(true)

    // Display variant drops the sentinel.
    expect(effectiveVisibleKeys(stored, providers).has(emptyProviderSentinelKey('nous'))).toBe(false)
  })
})

describe('toggleModelVisibility', () => {
  const providers = [provider('openai', ['gpt-a', 'gpt-b']), provider('nous', ['hermes-x', 'hermes-y'])]

  // Drive the handler the way the dialog does: feed each result back in as the
  // next `stored`, so the persisted set is what the next toggle starts from.
  const apply = (stored: Set<string> | null, slug: string, model: string) =>
    toggleModelVisibility(stored, providers, slug, model)

  it('records a hide-all sentinel when the last model of a provider is toggled off', () => {
    let stored: Set<string> | null = null
    stored = apply(stored, 'openai', 'gpt-a')
    stored = apply(stored, 'openai', 'gpt-b')

    expect(stored.has(emptyProviderSentinelKey('openai'))).toBe(true)
    expect(effectiveVisibleKeys(stored, providers).has(modelVisibilityKey('openai', 'gpt-a'))).toBe(false)
    expect(effectiveVisibleKeys(stored, providers).has(modelVisibilityKey('openai', 'gpt-b'))).toBe(false)
  })

  it('keeps a hidden provider hidden when a different provider is toggled (regression for #43485)', () => {
    // Hide ALL of nous — its sentinel is now stored.
    let stored: Set<string> | null = null
    stored = apply(stored, 'nous', 'hermes-x')
    stored = apply(stored, 'nous', 'hermes-y')
    expect(stored.has(emptyProviderSentinelKey('nous'))).toBe(true)

    // Toggle a model in another provider. nous must NOT snap back on.
    stored = apply(stored, 'openai', 'gpt-a')

    expect(stored.has(emptyProviderSentinelKey('nous'))).toBe(true)
    const visible = effectiveVisibleKeys(stored, providers)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(false)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-y'))).toBe(false)
  })

  it('clears only the toggled provider sentinel when a model is re-enabled', () => {
    let stored: Set<string> | null = new Set([emptyProviderSentinelKey('openai'), emptyProviderSentinelKey('nous')])

    stored = apply(stored, 'openai', 'gpt-a')

    expect(stored.has(emptyProviderSentinelKey('openai'))).toBe(false)
    expect(stored.has(emptyProviderSentinelKey('nous'))).toBe(true)
    const visible = effectiveVisibleKeys(stored, providers)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(false)
  })

  it('re-enabling one model of a hidden-all provider restores ONLY that model, not the curated defaults', () => {
    // openai hidden-all, nous untouched.
    let stored: Set<string> | null = new Set([emptyProviderSentinelKey('openai')])

    stored = apply(stored, 'openai', 'gpt-a')

    const visible = effectiveVisibleKeys(stored, providers)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
    // gpt-b is NOT restored — "you hid everything, you get back only what you re-enable".
    expect(visible.has(modelVisibilityKey('openai', 'gpt-b'))).toBe(false)
  })

  it('re-hiding the last re-enabled model re-adds the sentinel (full round-trip)', () => {
    let stored: Set<string> | null = new Set([emptyProviderSentinelKey('openai')])

    // Re-enable gpt-a (clears sentinel, set = {gpt-a}), then toggle it back off.
    stored = apply(stored, 'openai', 'gpt-a')
    expect(stored.has(emptyProviderSentinelKey('openai'))).toBe(false)
    stored = apply(stored, 'openai', 'gpt-a')

    expect(stored.has(emptyProviderSentinelKey('openai'))).toBe(true)
    expect(effectiveVisibleKeys(stored, providers).has(modelVisibilityKey('openai', 'gpt-a'))).toBe(false)
  })

  it('toggling from an empty (non-null) stored set adds the model without expanding defaults', () => {
    // Empty-but-not-null = "everything hidden". resolveVisibleKeys short-circuits to {}.
    const stored = new Set<string>()

    const next = apply(stored, 'openai', 'gpt-a')

    expect(next.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
    // No curated defaults were expanded for any provider.
    expect(next.has(modelVisibilityKey('openai', 'gpt-b'))).toBe(false)
    expect(next.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(false)
  })

  it('toggling off one default model from null stored keeps the rest of the curated defaults', () => {
    // null = "never customized": resolveVisibleKeys expands all defaults first.
    const next = apply(null, 'openai', 'gpt-a')

    expect(next.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(false)
    expect(next.has(modelVisibilityKey('openai', 'gpt-b'))).toBe(true)
    expect(next.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(true)
    // Other models remain, so no sentinel.
    expect(next.has(emptyProviderSentinelKey('openai'))).toBe(false)
  })

  it('tolerates a provider with zero models (defensive — dialog filters these out)', () => {
    const ps = [provider('empty', []), provider('openai', ['gpt-a'])]
    const next = toggleModelVisibility(new Set([modelVisibilityKey('openai', 'gpt-a')]), ps, 'empty', 'ghost')

    // No crash; the phantom key is recorded but no defaults are invented.
    expect([...next].some(k => k.startsWith('empty::') && !isProviderSentinel(k))).toBe(true)
    expect(next.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
  })
})

describe('resolveVisibleKeys', () => {
  const providers = [provider('openai', ['gpt-a', 'gpt-b']), provider('nous', ['hermes-x', 'hermes-y'])]

  it('returns the curated defaults verbatim for null stored', () => {
    expect(resolveVisibleKeys(null, providers)).toEqual(defaultVisibleKeys(providers))
  })

  it('returns an empty set for an empty (non-null) stored set', () => {
    expect([...resolveVisibleKeys(new Set(), providers)]).toEqual([])
  })
})

describe('compareModelFamilies', () => {
  const fam = (id: string) => ({ fastId: null, id })

  it('clusters families alphabetically with the newest version first', () => {
    const sorted = [
      fam('claude-sonnet-4-6'),
      fam('claude-opus-4'),
      fam('claude-opus-4-6'),
      fam('claude-fable-5'),
      fam('claude-haiku-4-5'),
      fam('claude-opus-4-8'),
      fam('claude-sonnet-4'),
      fam('claude-sonnet-5')
    ]
      .sort(compareModelFamilies)
      .map(f => f.id)

    // A bare major alias (opus-4) sorts BELOW its point releases — it's the
    // least specific/oldest of its line.
    expect(sorted).toEqual([
      'claude-fable-5',
      'claude-haiku-4-5',
      'claude-opus-4-8',
      'claude-opus-4-6',
      'claude-opus-4',
      'claude-sonnet-5',
      'claude-sonnet-4-6',
      'claude-sonnet-4'
    ])
  })

  it('is deterministic for equal ids and handles non-versioned names', () => {
    expect(compareModelFamilies(fam('gpt-5.2'), fam('gpt-5.2'))).toBe(0)
    expect([fam('mistral'), fam('llama'), fam('qwen')].sort(compareModelFamilies).map(f => f.id)).toEqual([
      'llama',
      'mistral',
      'qwen'
    ])
  })
})
