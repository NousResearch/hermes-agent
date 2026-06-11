import { describe, expect, it } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import { $knownProviders, $visibleModels, effectiveVisibleKeys, modelVisibilityKey, setVisibleModels } from './model-visibility'

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

    const visible = effectiveVisibleKeys(stored, [
      provider('local-ollama', ['qwen3:latest', 'llama3.2:latest'])
    ])

    expect(visible.has(modelVisibilityKey('local-ollama', 'qwen3:latest'))).toBe(true)
    expect(visible.has(modelVisibilityKey('local-ollama', 'llama3.2:latest'))).toBe(false)
  })

  it('keeps a known provider empty after its last model is hidden (no snap-back)', () => {
    // The user disabled every anthropic model down to the last one, so the
    // stored set no longer contains any `anthropic::` key — but anthropic is in
    // the known set because the user customized it. It must stay empty rather
    // than resurrecting all of anthropic's defaults (the reported bug).
    const stored = new Set([modelVisibilityKey('openai', 'gpt-5')])
    const known = new Set(['anthropic', 'openai'])

    const visible = effectiveVisibleKeys(
      stored,
      [
        provider('anthropic', ['haiku-4.5', 'sonnet-4.6', 'opus-4.8']),
        provider('openai', ['gpt-5'])
      ],
      known
    )

    expect(visible.has(modelVisibilityKey('anthropic', 'haiku-4.5'))).toBe(false)
    expect(visible.has(modelVisibilityKey('anthropic', 'sonnet-4.6'))).toBe(false)
    expect(visible.has(modelVisibilityKey('anthropic', 'opus-4.8'))).toBe(false)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-5'))).toBe(true)
  })

  it('still seeds a genuinely new provider absent from the known set', () => {
    // The user has customized anthropic, but `gemini` only appeared afterwards
    // (e.g. a new key). It was never customized, so it should show by default.
    const stored = new Set([modelVisibilityKey('anthropic', 'haiku-4.5')])
    const known = new Set(['anthropic'])

    const visible = effectiveVisibleKeys(
      stored,
      [
        provider('anthropic', ['haiku-4.5', 'opus-4.8']),
        provider('gemini', ['gemini-3-pro', 'gemini-3-flash'])
      ],
      known
    )

    // anthropic keeps its single explicit choice (opus stays hidden)...
    expect(visible.has(modelVisibilityKey('anthropic', 'opus-4.8'))).toBe(false)
    // ...while the brand-new gemini provider is seeded visible.
    expect(visible.has(modelVisibilityKey('gemini', 'gemini-3-pro'))).toBe(true)
    expect(visible.has(modelVisibilityKey('gemini', 'gemini-3-flash'))).toBe(true)
  })

  it('setVisibleModels unions known provider slugs across customizations', () => {
    $visibleModels.set(null)
    $knownProviders.set(null)

    setVisibleModels(new Set([modelVisibilityKey('anthropic', 'haiku-4.5')]), ['anthropic'])
    expect($knownProviders.get()).toEqual(new Set(['anthropic']))

    // A later customization made while only `openai` is on screen must not drop
    // anthropic from the known set.
    setVisibleModels(new Set([modelVisibilityKey('openai', 'gpt-5')]), ['openai'])
    expect($knownProviders.get()).toEqual(new Set(['anthropic', 'openai']))

    // Omitting the slugs (non-customization writes) leaves the known set intact.
    setVisibleModels(new Set())
    expect($knownProviders.get()).toEqual(new Set(['anthropic', 'openai']))
  })
})
