import { describe, expect, it } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import {
  effectiveVisibleKeys,
  disableProvider,
  enableProvider,
  modelVisibilityKey,
  toggleProviderVisibility,
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

  it('respects tombstone — disabled provider does not get defaults re-added', () => {
    const p = provider('anthropic', ['opus-4', 'sonnet-4'])
    const keys = new Set([modelVisibilityKey('local', 'qwen')])

    disableProvider(p, keys)

    expect(keys.has(modelVisibilityKey('anthropic', 'opus-4'))).toBe(false)
    expect(keys.has(modelVisibilityKey('anthropic', 'sonnet-4'))).toBe(false)
    expect(keys.has('anthropic::')).toBe(true)

    // effectiveVisibleKeys should NOT re-add Anthropic defaults because of the tombstone.
    const visible = effectiveVisibleKeys(keys, [p, provider('local', ['qwen'])])
    expect(visible.has(modelVisibilityKey('anthropic', 'opus-4'))).toBe(false)
    expect(visible.has(modelVisibilityKey('anthropic', 'sonnet-4'))).toBe(false)
  })

  it('enableProvider adds all models back and removes tombstone', () => {
    const p = provider('google', ['gemini-flash', 'gemini-pro'])
    const keys = new Set(['google::']) // tombstone only

    enableProvider(p, keys)

    expect(keys.has(modelVisibilityKey('google', 'gemini-flash'))).toBe(true)
    expect(keys.has(modelVisibilityKey('google', 'gemini-pro'))).toBe(true)
  })

  it('toggleProviderVisibility: first call disables, second re-enables', () => {
    const p = provider('openai', ['gpt-4o', 'gpt-5'])
    const keys = new Set([
      modelVisibilityKey('openai', 'gpt-4o'),
      modelVisibilityKey('openai', 'gpt-5'),
    ])

    // First toggle: should disable all.
    toggleProviderVisibility(p, keys)
    expect(keys.has(modelVisibilityKey('openai', 'gpt-4o'))).toBe(false)
    expect(keys.has(modelVisibilityKey('openai', 'gpt-5'))).toBe(false)
    expect(keys.has('openai::')).toBe(true)

    // Second toggle: should re-enable all.
    toggleProviderVisibility(p, keys)
    expect(keys.has(modelVisibilityKey('openai', 'gpt-4o'))).toBe(true)
    expect(keys.has(modelVisibilityKey('openai', 'gpt-5'))).toBe(true)
    expect(keys.has('openai::')).toBe(false)
  })

  it('-fast siblings collapse into families correctly for provider toggle', () => {
    const p = provider('anthropic', ['opus-4', 'opus-4-fast', 'sonnet-4'])
    const keys = new Set([modelVisibilityKey('anthropic', 'opus-4'), modelVisibilityKey('anthropic', 'sonnet-4')])

    disableProvider(p, keys)
    expect(keys.has(modelVisibilityKey('anthropic', 'opus-4'))).toBe(false)
    expect(keys.has(modelVisibilityKey('anthropic', 'sonnet-4'))).toBe(false)
    expect(keys.has('anthropic::')).toBe(true)

    enableProvider(p, keys)
    // opus-4 and sonnet-4 are the family base ids (opus-4-fast collapsed under opus-4)
    expect(keys.has(modelVisibilityKey('anthropic', 'opus-4'))).toBe(true)
    expect(keys.has(modelVisibilityKey('anthropic', 'sonnet-4'))).toBe(true)
  })
})
