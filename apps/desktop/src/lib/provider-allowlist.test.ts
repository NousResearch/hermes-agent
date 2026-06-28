import { describe, expect, it } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import { filterPickerProviders, isPickerVisibleProvider } from './provider-allowlist'

const provider = (slug: string): ModelOptionProvider => ({ name: slug, slug, models: ['m'] })

describe('provider-allowlist', () => {
  it('keeps the APEX-NODES.COM managed relay and custom BYOK endpoints', () => {
    // Managed relay is a named custom provider (custom_provider_slug('Apex-nodes.com')).
    expect(isPickerVisibleProvider('custom:apex-nodes.com')).toBe(true)
    // A user's own OpenAI-compatible endpoint (bare or named).
    expect(isPickerVisibleProvider('custom')).toBe(true)
    expect(isPickerVisibleProvider('custom:my-local')).toBe(true)
  })

  it('keeps domestic (国产) providers for BYOK', () => {
    for (const slug of [
      'deepseek',
      'zai',
      'kimi-coding',
      'kimi-coding-cn',
      'alibaba',
      'alibaba-coding-plan',
      'qwen-oauth',
      'minimax',
      'minimax-oauth',
      'minimax-cn',
      'stepfun',
      'xiaomi',
      'tencent-tokenhub'
    ]) {
      expect(isPickerVisibleProvider(slug)).toBe(true)
    }
  })

  it('hides GitHub Copilot and every foreign provider', () => {
    for (const slug of [
      'copilot',
      'copilot-acp',
      'openai',
      'openai-api',
      'openai-codex',
      'anthropic',
      'xai',
      'xai-oauth',
      'gemini',
      'google-gemini-cli',
      'mistral',
      'cohere',
      'openrouter',
      'nous',
      'bedrock',
      'azure-foundry',
      'nvidia',
      'huggingface',
      'ollama-cloud',
      'groq',
      'together',
      'fireworks'
    ]) {
      expect(isPickerVisibleProvider(slug)).toBe(false)
    }
  })

  it('is case-insensitive and rejects empty slugs', () => {
    expect(isPickerVisibleProvider('DeepSeek')).toBe(true)
    expect(isPickerVisibleProvider('Custom:Apex-Nodes.com')).toBe(true)
    expect(isPickerVisibleProvider('')).toBe(false)
    expect(isPickerVisibleProvider('   ')).toBe(false)
  })

  it('filters a provider list while preserving order', () => {
    const input = [
      provider('copilot'),
      provider('deepseek'),
      provider('openai'),
      provider('custom:apex-nodes.com'),
      provider('zai'),
      provider('anthropic')
    ]

    expect(filterPickerProviders(input).map(p => p.slug)).toEqual([
      'deepseek',
      'custom:apex-nodes.com',
      'zai'
    ])
  })
})
