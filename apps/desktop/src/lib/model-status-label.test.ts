import { describe, expect, it } from 'vitest'

import {
  currentPickerSelection,
  displayModelName,
  formatModelStatusLabel,
  modelDisplayParts
} from './model-status-label'
import { reasoningEffortLabel } from './reasoning-effort'

describe('model-status-label', () => {
  it('formats display names consistently', () => {
    expect(displayModelName('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8')
    expect(displayModelName('openai/gpt-5.5-fast')).toBe('GPT-5.5')
    expect(displayModelName('deepseek/deepseek-v4-pro-thinking')).toBe('Deepseek V4 Pro')
    expect(displayModelName('openai/gpt-5.5')).toBe('GPT-5.5')
  })

  it('keeps an inference-router route prefix ahead of the vendor path', () => {
    // Router-provided ids carry the route in the id itself; the vendor path
    // after it is still noise, but the route must not be swallowed with it.
    expect(displayModelName('CC : deepseek/deepseek-v4-pro')).toBe('CC : Deepseek V4 Pro')
    expect(displayModelName('CC : moonshotai/Kimi-K3')).toBe('CC : Kimi K3')
    expect(displayModelName('CC : z-ai/glm-5.3-flash')).toBe('CC : Glm 5.3 Flash')
    expect(modelDisplayParts('CC : tencent/hy4-preview')).toEqual({ name: 'CC : Hy4', tag: 'Preview' })
    expect(displayModelName('OC : deepseek-v4-pro')).toBe('OC : Deepseek V4 Pro')
    expect(displayModelName('oMLX : Qwen3.8-27B-8bit')).toBe('OMLX : Qwen3.8 27B 8bit')
    // Ollama-style tags use a bare colon and are not a route prefix.
    expect(displayModelName('deepseek-v4-flash:0731')).toBe('Deepseek V4 Flash:0731')
    expect(displayModelName('OL : deepseek-v4-flash:0731')).toBe('OL : Deepseek V4 Flash:0731')
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
