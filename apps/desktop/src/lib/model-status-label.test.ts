import { describe, expect, it } from 'vitest'

import { currentPickerSelection, displayModelName, formatModelPillLabel, modelDisplayParts } from './model-status-label'

describe('model-status-label', () => {
  it('strips trailing date-pin snapshots and dots hyphenated Anthropic versions', () => {
    expect(displayModelName('claude-opus-4-5-20251101')).toBe('Opus 4.5')
    expect(displayModelName('anthropic/claude-haiku-4-5-20251001')).toBe('Haiku 4.5')
    expect(displayModelName('claude-fable-5-1')).toBe('Fable 5.1')
  })

  it('renders the Anthropic 1M-context route suffix as a tag, never raw brackets', () => {
    expect(modelDisplayParts('claude-sonnet-5[1m]')).toEqual({ name: 'Sonnet 5', tag: '1M' })
    expect(modelDisplayParts('claude-fable-5-1[1m]')).toEqual({ name: 'Fable 5.1', tag: '1M' })
    expect(displayModelName('claude-opus-5[1m]')).not.toContain('[')
  })

  it('renders local GGUF ids as a clean name with a quant tag', () => {
    expect(modelDisplayParts('Qwen3.6-27B-UD-Q4_K_XL')).toEqual({ name: 'Qwen3.6 27B', tag: 'Q4' })
    expect(modelDisplayParts('Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL')).toEqual({
      name: 'Nemotron 3 Nano 30B A3B',
      tag: 'Q4'
    })
    expect(modelDisplayParts('Qwen3-4B-Instruct-2507-UD-Q8_K_XL')).toEqual({ name: 'Qwen3 4B', tag: 'Q8' })
    expect(modelDisplayParts('some-model-Q6_K')).toEqual({ name: 'Some Model', tag: 'Q6' })
    // Cloud ids keep their existing behavior.
    expect(modelDisplayParts('anthropic/claude-opus-4.8-fast').tag).toBe('Fast')
  })

  it('keeps the vendor casing the model id does not carry (#85849)', () => {
    expect(displayModelName('glm-5.2')).toBe('GLM 5.2')
    expect(displayModelName('zai-org/glm-5.1')).toBe('GLM 5.1')
    expect(displayModelName('deepseek-v4-flash')).toBe('DeepSeek V4 Flash')
    expect(displayModelName('minimax/minimax-01')).toBe('MiniMax 01')
    expect(displayModelName('xiaomi/mimo-v2.5')).toBe('MiMo V2.5')
    expect(displayModelName('ernie-5.1')).toBe('ERNIE 5.1')
    expect(displayModelName('baai/bge-m3')).toBe('BGE M3')
    expect(displayModelName('openai')).toBe('OpenAI')
  })

  it('capitalises parameter counts the way vendors write them (#85849)', () => {
    expect(displayModelName('qwen3-32b')).toBe('Qwen3 32B')
    expect(displayModelName('qwen/qwen3.5-35b-a3b')).toBe('Qwen3.5 35B A3B')
    expect(displayModelName('meta/llama-3.1-8b-instruct')).toBe('Llama 3.1 8B Instruct')
    expect(displayModelName('llama-3.1-8b-instruct-fp8')).toBe('Llama 3.1 8B Instruct FP8')
    expect(displayModelName('gemma-4-26b-a4b-it')).toBe('Gemma 4 26B A4B IT')
    expect(displayModelName('nemotron-nano-12b-v2-vl')).toBe('Nemotron Nano 12B V2 VL')
  })

  it('title-cases gemini names like every other branch (#85849)', () => {
    expect(displayModelName('gemini-2.5-pro')).toBe('Gemini 2.5 Pro')
    expect(displayModelName('gemini-2.0-flash')).toBe('Gemini 2.0 Flash')
    expect(displayModelName('google/gemini-2.5-flash-lite')).toBe('Gemini 2.5 Flash Lite')
  })

  it('keeps the model pill to name + Fast; the effort lives on its own pill', () => {
    expect(formatModelPillLabel('openai/gpt-5.5', { fastMode: true })).toBe('GPT-5.5 · Fast')
    expect(formatModelPillLabel('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8 Fast')
    expect(formatModelPillLabel('openai/gpt-5.5')).toBe('GPT-5.5')
    expect(formatModelPillLabel('')).toBe('No model')
  })

  it('keeps the variant tag in the display name so distinct ids never collapse (#88597)', () => {
    expect(displayModelName('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8 Fast')
    expect(displayModelName('deepseek/deepseek-v4-pro-thinking')).toBe('DeepSeek V4 Pro Thinking')
    expect(displayModelName('gpt-5.5-preview')).toBe('GPT-5.5 Preview')
    expect(displayModelName('claude-opus-5')).toBe('Opus 5')
    // A base model and its variant must NEVER share a display label.
    expect(displayModelName('claude-opus-5')).not.toBe(displayModelName('claude-opus-5-thinking'))
    // The quant/contextWindow tags ride along the same way.
    expect(displayModelName('Qwen3.6-27B-UD-Q4_K_XL')).toBe('Qwen3.6 27B Q4')
    expect(displayModelName('claude-sonnet-5[1m]')).toBe('Sonnet 5 1M')
  })

  describe('currentPickerSelection', () => {
    const store = { model: 'opus', provider: 'anthropic' }
    const options = { model: 'hermes-4', provider: 'nous' }

    it('prefers the sticky composer pick over the profile default pre-session', () => {
      expect(currentPickerSelection(store, options)).toEqual(store)
    })

    it('falls back to options when the store is empty', () => {
      expect(currentPickerSelection({ model: '', provider: '' }, options)).toEqual(options)
    })

    it('uses the complete options pair instead of mixing a partial store selection', () => {
      expect(currentPickerSelection({ model: 'opus', provider: '' }, options)).toEqual(options)
    })

    it('falls back to the store while options are still loading', () => {
      expect(currentPickerSelection(store, undefined)).toEqual(store)
    })
  })
})
