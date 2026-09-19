import { describe, expect, it } from 'vitest'

import { currentPickerSelection, displayModelName, formatModelPillLabel, modelDisplayParts, modelPillParts, providerSegment } from './model-status-label'
import { reasoningEffortLabel } from './reasoning-effort'

describe('model-status-label', () => {
  it('formats display names consistently', () => {
    expect(displayModelName('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8')
    expect(displayModelName('openai/gpt-5.5-fast')).toBe('GPT-5.5')
    expect(displayModelName('deepseek/deepseek-v4-pro-thinking')).toBe('Deepseek V4 Pro')
    expect(displayModelName('openai/gpt-5.5')).toBe('GPT-5.5')
  })

  it('strips trailing date-pin snapshots from the display name', () => {
    expect(displayModelName('claude-opus-4-5-20251101')).toBe('Opus 4 5')
    expect(displayModelName('anthropic/claude-haiku-4-5-20251001')).toBe('Haiku 4 5')
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

  it('maps reasoning effort to compact labels', () => {
    expect(reasoningEffortLabel('high')).toBe('High')
    expect(reasoningEffortLabel('xhigh')).toBe('XHigh')
    expect(reasoningEffortLabel('max')).toBe('Max')
    expect(reasoningEffortLabel('ultra')).toBe('Ultra')
    expect(reasoningEffortLabel('')).toBe('')
  })

  it('keeps the model pill to name + Fast; the effort lives on its own pill', () => {
    expect(formatModelPillLabel('openai/gpt-5.5', { fastMode: true })).toBe('GPT-5.5 · Fast')
    expect(formatModelPillLabel('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8 · Fast')
    expect(formatModelPillLabel('openai/gpt-5.5')).toBe('GPT-5.5')
    expect(formatModelPillLabel('')).toBe('No model')
  })

  describe('the provider segment', () => {
    it('shows the provider beside the model name when it adds information', () => {
      expect(modelPillParts('gpt-5.5', { provider: 'openai' })).toEqual({
        fast: false,
        name: 'GPT-5.5',
        provider: 'openai'
      })
      expect(modelPillParts('gpt-5.5', { fastMode: true, provider: 'openai' }).provider).toBe('openai')
    })

    it('hides the provider when the model id already carries the same prefix', () => {
      expect(modelPillParts('openai/gpt-5.5', { provider: 'openai' }).provider).toBe('')
      // Case-insensitive: the catalog slug and the id prefix may differ only in case.
      expect(modelPillParts('OpenAI/gpt-5.5', { provider: 'openai' }).provider).toBe('')
    })

    it('keeps the provider when the id prefix names a different upstream', () => {
      // Relay rows carry the upstream vendor in the deeper path, so the top-level
      // prefix still adds a distinct fact — both halves are information.
      expect(
        modelPillParts('openrouter/anthropic/claude-opus-4.8', { provider: 'openrouter' }).provider
      ).toBe('openrouter')
    })

    it('does not hide a provider that is only a substring of the id prefix', () => {
      expect(modelPillParts('openai-custom/gpt-6', { provider: 'openai' }).provider).toBe('openai')
    })

    it('hides the provider when none is known', () => {
      expect(modelPillParts('gpt-5.5').provider).toBe('')
      expect(modelPillParts('gpt-5.5', { provider: '   ' }).provider).toBe('')
      expect(providerSegment('gpt-5.5', undefined)).toBe('')
    })

    it('handles provider slugs carrying colons or slashes', () => {
      expect(providerSegment('gpt-5.5', 'custom:corp/gateway')).toBe('custom:corp/gateway')
    })

    it('keeps the no-model placeholder free of a provider', () => {
      expect(modelPillParts('', { provider: 'nous' })).toEqual({
        fast: false,
        name: 'No model',
        provider: ''
      })
      expect(modelPillParts('', { fastMode: true, provider: 'nous' })).toEqual({
        fast: false,
        name: 'No model',
        provider: ''
      })
    })

    it('agrees with the string form on Fast', () => {
      expect(modelPillParts('anthropic/claude-opus-4.8-fast').fast).toBe(true)
      expect(modelPillParts('openai/gpt-5.5', { fastMode: true }).fast).toBe(true)
      expect(modelPillParts('openai/gpt-5.5').fast).toBe(false)
    })
  })

  describe('currentPickerSelection', () => {
    const store = { model: 'opus', provider: 'anthropic' }
    const options = { model: 'hermes-4', provider: 'nous' }

    it('prefers the sticky composer pick over the profile default pre-session', () => {
      expect(currentPickerSelection(store, options)).toEqual(store)
    })

    it('keeps the SessionView selection when a stale options response disagrees', () => {
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
