import { describe, expect, it } from 'vitest'

import { modelBrand } from './model-brand'

describe('modelBrand', () => {
  it.each([
    ['gpt-6-astra', 'openai'],
    ['openai/gpt-5.2-codex', 'openai'],
    ['openai/o3', 'openai'],
    ['chatgpt-4o-latest', 'openai'],
    ['anthropic/claude-sonnet-4.5', 'anthropic'],
    ['us.anthropic.claude-3-5-sonnet-20241022-v2:0', 'anthropic'],
    ['glm-5.2', 'zai'],
    ['z-ai/glm-4.5-air', 'zai'],
    ['zai-org/GLM-4.7', 'zai'],
    ['google/gemini-2.5-pro', 'google'],
    ['gemma3:27b', 'google'],
    ['deepseek-ai/DeepSeek-R1', 'deepseek'],
    ['qwen3:30b', 'qwen'],
    ['Qwen/QwQ-32B', 'qwen'],
    ['x-ai/grok-4', 'xai'],
    ['mistralai/Mistral-Small-3.2', 'mistral'],
    ['codestral-latest', 'mistral'],
    ['devstral-2', 'mistral'],
    ['meta-llama/Llama-3.3-70B-Instruct', 'meta'],
    ['NousResearch/Hermes-4-70B', 'nous'],
    // The model family wins over the hosting provider or GGUF repository.
    ['openrouter/anthropic/claude-opus-4.1', 'anthropic'],
    ['unsloth/GLM-4.7-GGUF', 'zai'],
    ['  GPT-5.2  ', 'openai']
  ])('resolves %s to its model author %s', (model, brand) => {
    expect(modelBrand(model)).toBe(brand)
  })

  it.each([null, undefined, '', 'custom-model', 'my-gpt-proxy', 'openrouter/auto', 'openai/unknown-model'])(
    'does not invent a brand for %s',
    model => {
      expect(modelBrand(model)).toBe('unknown')
    }
  )
})
