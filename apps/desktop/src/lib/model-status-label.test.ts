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

  it('keeps the model pill to name + Fast; the effort lives on its own pill', () => {
    expect(formatModelPillLabel('openai/gpt-5.5', { fastMode: true })).toBe('GPT-5.5 · Fast')
    expect(formatModelPillLabel('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8 · Fast')
    expect(formatModelPillLabel('openai/gpt-5.5')).toBe('GPT-5.5')
    expect(formatModelPillLabel('')).toBe('No model')
  })

  it('renders each id token by class: sizes upper-case, versions dotted, brands cased', () => {
    // Parameter counts and active-param markers are always upper-case B.
    expect(displayModelName('qwen/qwen3-235b-a22b')).toBe('Qwen3 235B A22B')
    expect(displayModelName('meta-llama/llama-3.3-70b-instruct')).toBe('Llama 3.3 70B Instruct')
    expect(displayModelName('mistralai/mixtral-8x22b')).toBe('Mixtral 8x22B')
    // Consecutive bare numbers are one dotted version; hyphenated brands keep
    // the hyphen before it.
    expect(displayModelName('zai/glm-5-1')).toBe('GLM-5.1')
    expect(displayModelName('openai/gpt-6-terra')).toBe('GPT-6 Terra')
    expect(displayModelName('openai/gpt-oss-120b')).toBe('GPT OSS 120B')
    expect(displayModelName('openai/gpt-5.4-mini')).toBe('GPT-5.4 mini')
    // Brand casing and letter-versions.
    expect(displayModelName('deepseek/deepseek-v4-pro')).toBe('DeepSeek V4 Pro')
    expect(displayModelName('moonshotai/kimi-k2.5')).toBe('Kimi K2.5')
    expect(displayModelName('google/gemini-3-pro')).toBe('Gemini 3 Pro')
    expect(displayModelName('nvidia/llama-3.1-nemotron-70b-fp8')).toBe('Llama 3.1 Nemotron 70B FP8')
  })

  it('never reinterprets a version and keeps author casing', () => {
    // A brand that already carries digits does not absorb the next number.
    expect(displayModelName('qwen3-5-122b-a10b')).toBe('Qwen3 5 122B A10B')
    // A trailing 4-digit snapshot pin is dropped, not merged into the version.
    expect(displayModelName('deepseek/deepseek-r1-0528')).toBe('DeepSeek R1')
    // Hand-cased tokens survive untouched.
    expect(displayModelName('Qwen3-30B-A3B-MoE')).toBe('Qwen3 30B A3B MoE')
    expect(displayModelName('NousResearch/Hermes-4-405B')).toBe('Hermes 4 405B')
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
