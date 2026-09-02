import { describe, expect, it } from 'vitest'

import {
  currentPickerSelection,
  displayModelName,
  formatModelStatusLabel,
  modelDisplayParts,
  modelStatusLabelParts
} from './model-status-label'
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

  it('appends fast + effort session state to the status label', () => {
    expect(formatModelStatusLabel('openai/gpt-5.5', { fastMode: true, reasoningEffort: 'high' })).toBe(
      'GPT-5.5 · Fast High'
    )
  })

  it('shows the provider between the model name and the state', () => {
    expect(formatModelStatusLabel('gpt-5.5', { provider: 'openai', reasoningEffort: 'medium' })).toBe(
      'GPT-5.5 · openai · Med'
    )
    expect(formatModelStatusLabel('qwen3-max', { provider: 'custom:my_pool', reasoningEffort: '' })).toBe(
      'Qwen3 Max · custom:my_pool · Med'
    )
  })

  it('hides the provider when the model id already carries the same prefix', () => {
    expect(formatModelStatusLabel('openai/gpt-5.5', { provider: 'openai', reasoningEffort: 'high' })).toBe(
      'GPT-5.5 · High'
    )
    // Case-insensitive: the catalog slug and the id prefix may differ only in case.
    expect(formatModelStatusLabel('OpenAI/gpt-5.5', { provider: 'openai' })).toBe('GPT-5.5 · Med')
  })

  it('keeps the provider when the id prefix names a different upstream', () => {
    // OpenRouter ids carry the upstream vendor prefix; the row's provider is
    // still openrouter — both halves are information, so both show.
    expect(formatModelStatusLabel('openrouter/anthropic/claude-opus-4.8', { provider: 'openrouter' })).toBe(
      'Opus 4.8 · openrouter · Med'
    )
  })

  it('hides the provider when none is known', () => {
    expect(formatModelStatusLabel('gpt-5.5', { reasoningEffort: 'medium' })).toBe('GPT-5.5 · Med')
    expect(formatModelStatusLabel('gpt-5.5', { provider: '  ', reasoningEffort: 'medium' })).toBe('GPT-5.5 · Med')
  })

  it('does not deduplicate when provider is only a substring of model prefix', () => {
    // 'openai-custom/gpt-6' does NOT have prefix 'openai/', so 'openai' should not be stripped
    expect(formatModelStatusLabel('openai-custom/gpt-6', { provider: 'openai' })).toBe('Gpt-6 · openai · Med')
  })

  it('handles provider names with custom colons or slashes', () => {
    expect(modelStatusLabelParts('gpt-5.5', { provider: 'custom:corp/gateway' })).toEqual({
      meta: 'Med',
      name: 'GPT-5.5',
      provider: 'custom:corp/gateway'
    })
  })

  it('splits the label into styled parts for the composer pill', () => {
    expect(
      modelStatusLabelParts('gpt-5.5', { fastMode: true, provider: 'openai', reasoningEffort: 'high' })
    ).toEqual({ meta: 'Fast High', name: 'GPT-5.5', provider: 'openai' })
    // The dedupe lives in the parts, so both renderers agree.
    expect(modelStatusLabelParts('openai/gpt-5.5', { provider: 'openai' }).provider).toBe('')
    expect(modelStatusLabelParts('openai/gpt-5.5', { provider: 'nous' }).provider).toBe('nous')
    expect(modelStatusLabelParts('', { provider: 'nous' })).toEqual({ meta: '', name: 'No model', provider: '' })
  })

  it('falls back to the profile default effort, then to medium', () => {
    expect(formatModelStatusLabel('openai/gpt-5.5', { reasoningEffort: 'medium' })).toBe('GPT-5.5 · Med')
    expect(formatModelStatusLabel('openai/gpt-5.5')).toBe('GPT-5.5 · Med')
    // No session-level effort → the configured profile default is advertised,
    // not Hermes' built-in medium.
    expect(formatModelStatusLabel('openai/gpt-5.5', { defaultEffort: 'high' })).toBe('GPT-5.5 · High')
    // An explicit session effort still wins over the profile default.
    expect(formatModelStatusLabel('openai/gpt-5.5', { defaultEffort: 'high', reasoningEffort: 'low' })).toBe(
      'GPT-5.5 · Low'
    )
  })

  it('returns just the placeholder name when there is no model', () => {
    expect(formatModelStatusLabel('')).toBe('No model')
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
