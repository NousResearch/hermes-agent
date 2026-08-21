import { describe, expect, it } from 'vitest'

import { currentPickerSelection, displayModelName, formatModelStatusLabel } from './model-status-label'
import { reasoningEffortLabel } from './reasoning-effort'

describe('model-status-label', () => {
  it('formats display names consistently', () => {
    expect(displayModelName('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8')
    expect(displayModelName('openai/gpt-5.5-fast')).toBe('GPT-5.5')
    expect(displayModelName('deepseek/deepseek-v4-pro-thinking')).toBe('DeepSeek V4 Pro')
    expect(displayModelName('openai/gpt-5.5')).toBe('GPT-5.5')
  })

  it('strips trailing date-pin snapshots from the display name', () => {
    expect(displayModelName('claude-opus-4-5-20251101')).toBe('Opus 4 5')
    expect(displayModelName('anthropic/claude-haiku-4-5-20251001')).toBe('Haiku 4 5')
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

  it('keeps vendor casing that the model id does not carry', () => {
    expect(displayModelName('glm-5.2')).toBe('GLM 5.2')
    expect(displayModelName('zai-org/glm-5.1')).toBe('GLM 5.1')
    expect(displayModelName('deepseek-v4-flash')).toBe('DeepSeek V4 Flash')
    expect(displayModelName('minimax/minimax-01')).toBe('MiniMax 01')
    expect(displayModelName('xiaomi/mimo-v2.5')).toBe('MiMo V2.5')
    expect(displayModelName('ernie-5.1')).toBe('ERNIE 5.1')
    expect(displayModelName('baai/bge-m3')).toBe('BGE M3')
  })

  it('capitalises parameter counts the way vendors write them', () => {
    expect(displayModelName('qwen3-32b')).toBe('Qwen3 32B')
    expect(displayModelName('qwen/qwen3.5-35b-a3b')).toBe('Qwen3.5 35B A3B')
    expect(displayModelName('meta/llama-3.1-8b-instruct-fp8')).toBe('Llama 3.1 8B Instruct FP8')
  })

  it('title-cases gemini names like every other branch', () => {
    expect(displayModelName('gemini-2.5-pro')).toBe('Gemini 2.5 Pro')
    expect(displayModelName('gemini-2.0-flash')).toBe('Gemini 2.0 Flash')
    expect(displayModelName('google/gemini-2.5-flash-lite')).toBe('Gemini 2.5 Flash Lite')
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
