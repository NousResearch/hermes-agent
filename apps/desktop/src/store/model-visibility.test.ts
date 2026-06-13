import { describe, expect, it } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import {
  effectiveVisibleKeys,
  emptyProviderSentinelKey,
  isProviderSentinel,
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
    const stored = new Set([
      emptyProviderSentinelKey('nous'),
      modelVisibilityKey('ollama', 'qwen3:latest')
    ])

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

  it('sentinel key helper produces correct format', () => {
    expect(emptyProviderSentinelKey('openai')).toBe('openai::')
    expect(isProviderSentinel('openai::')).toBe(true)
    expect(isProviderSentinel('openai::gpt-4o')).toBe(false)
  })
})
