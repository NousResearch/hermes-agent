import { expect, it } from 'vitest'

import { resolveModelBrand } from './model-brand-icon'

const PRESET_MODELS = [
  'deepseek/deepseek-v4.1-flash',
  'openai/gpt-5.6',
  'anthropic/claude-sonnet-4.5',
  'google/gemini-3.8-flash',
  'nousresearch/hermes-4-405b',
  'meta-llama/llama-4:free'
]

it('gives every preset model its own maker mark', () => {
  const brands = PRESET_MODELS.map(model => resolveModelBrand(model).name)

  expect(new Set(brands).size).toBe(PRESET_MODELS.length)
})

it('reads the maker from the id first, then the model name, then the provider, and never throws', () => {
  expect(resolveModelBrand('gpt-realtime', 'openai').name).toBe('OpenAI')
  expect(resolveModelBrand('some-custom-model', 'anthropic').name).toBe('Claude')
  expect(resolveModelBrand(undefined, null, '').name).toBe('Model')
})
