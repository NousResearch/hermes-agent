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

  it('distinguishes the deepseek-flash alias from its deepseek-v4.1-flash sibling (#118083)', () => {
    // models.dev carries both ids for the provider: `deepseek-flash` (alias)
    // and `deepseek-v4.1-flash` (full id). Two distinct ids must never render
    // as near-identical tagless rows the user reads as one model listed twice.
    // The `-flash` variant tag splits the pair the same way `-fast` splits
    // `…-4.8` vs `…-4.8-fast`, and the vendor word fixes the casing gap that
    // made the alias read "DeepSeek" while its sibling read "Deepseek".
    expect(modelDisplayParts('deepseek-flash')).toEqual({ name: 'DeepSeek', tag: 'Flash' })
    expect(modelDisplayParts('deepseek-v4.1-flash')).toEqual({ name: 'DeepSeek V4.1', tag: 'Flash' })
    // Non-flash siblings stay tagless and distinct from their flash variant.
    expect(modelDisplayParts('deepseek-v4.1')).toEqual({ name: 'DeepSeek V4.1', tag: '' })
  })

  it('spells vendor words the way the vendor does, not naive title-case (#118083)', () => {
    expect(displayModelName('deepseek-chat')).toBe('DeepSeek Chat')
    expect(displayModelName('zai/glm-5.2-air')).toBe('GLM 5.2 Air')
    expect(displayModelName('qwen3.6-vl-plus')).toBe('Qwen3.6 Vl Plus')
  })

  it('keeps the model pill to name + Fast; the effort lives on its own pill', () => {
    expect(formatModelPillLabel('openai/gpt-5.5', { fastMode: true })).toBe('GPT-5.5 · Fast')
    expect(formatModelPillLabel('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8 · Fast')
    expect(formatModelPillLabel('openai/gpt-5.5')).toBe('GPT-5.5')
    expect(formatModelPillLabel('')).toBe('No model')
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
