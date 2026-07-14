import { describe, expect, it } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import {
  collapseModelFamilies,
  effectiveVisibleKeys,
  modelVisibilityKey
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

    const visible = effectiveVisibleKeys(stored, [
      provider('local-ollama', ['qwen3:latest', 'llama3.2:latest'])
    ])

    expect(visible.has(modelVisibilityKey('local-ollama', 'qwen3:latest'))).toBe(true)
    expect(visible.has(modelVisibilityKey('local-ollama', 'llama3.2:latest'))).toBe(false)
  })

  it('drop non-string model entries instead of crashing on .toLowerCase()', () => {
    // Backend occasionally delivers objects/undefined in `models` instead of
    // strings. `collapseModelFamilies` (the shared boundary) must normalize
    // them away so callers never hit the invalid-data path.
    const mixed = ['claude-sonnet-4.6', { id: 'oops' }, undefined, 42, 'gpt-4o'] as unknown[]

    const families = collapseModelFamilies(mixed)

    expect(families.map(f => f.id)).toEqual(['claude-sonnet-4.6', 'gpt-4o'])
  })

  it('keeps working when a provider mixes string and object model entries', () => {
    const providers = [
      provider('copilot', ['claude-sonnet-4.6', { name: 'not-a-string' }, undefined, 'gpt-4o'] as unknown as string[])
    ]

    const visible = effectiveVisibleKeys(null, providers)

    expect(visible.has(modelVisibilityKey('copilot', 'claude-sonnet-4.6'))).toBe(true)
    expect(visible.has(modelVisibilityKey('copilot', 'gpt-4o'))).toBe(true)
  })
})
