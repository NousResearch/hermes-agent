import { describe, expect, it } from 'vitest'

import {
  currentPickerSelection,
  displayModelName,
  formatModelPillLabel,
  modelDisplayParts,
  modelListLabeler,
  providerDisplayName
} from './model-status-label'
import { reasoningEffortLabel } from './reasoning-effort'

describe('model-status-label', () => {
  it('uses the canonical provider label for xAI OAuth model picks', () => {
    expect(providerDisplayName('xai-oauth')).toBe('xAI Grok')
    expect(providerDisplayName('xai')).toBe('xAI')
    expect(providerDisplayName('openai-codex')).toBe('ChatGPT or Codex Subscription')
  })
  it('formats display names consistently', () => {
    expect(displayModelName('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8')
    expect(displayModelName('openai/gpt-5.5-fast')).toBe('GPT-5.5')
    expect(displayModelName('deepseek/deepseek-v4-pro-thinking')).toBe('Deepseek V4 Pro')
    expect(displayModelName('deepseek/deepseek-flash')).toBe('DeepSeek V4.1 Flash')
    expect(displayModelName('openai/gpt-5.5')).toBe('GPT-5.5')
  })

  describe('modelListLabeler', () => {
    const labelKey = ({ name, tag }: { name: string; tag: string }) => `${name}\u0000${tag}`.toLowerCase()

    it('never gives two distinct ids in one list the same label (#118083)', () => {
      // OpenCode Go lists the canonical Flash id beside its versioned id. Their
      // labels differ only in case, and would match exactly once the vendor is
      // spelled the same way on both.
      const ids = ['deepseek-flash', 'deepseek-v4.1-flash', 'deepseek-v4-pro', 'minimax-m3']
      const labelFor = modelListLabeler(ids)

      expect(new Set(ids.map(id => labelKey(labelFor(id)))).size).toBe(ids.length)
    })

    it('leaves rows that do not collide exactly as modelDisplayParts renders them', () => {
      const ids = ['deepseek-flash', 'deepseek-v4.1-flash', 'deepseek-v4-pro', 'claude-opus-4.8-fast']
      const labelFor = modelListLabeler(ids)

      for (const id of ['deepseek-v4-pro', 'claude-opus-4.8-fast']) {
        expect(labelFor(id)).toEqual(modelDisplayParts(id))
      }
    })

    it('keeps the display name on a colliding row and tells it apart by its id', () => {
      const labelFor = modelListLabeler(['deepseek-flash', 'deepseek-v4.1-flash'])

      for (const id of ['deepseek-flash', 'deepseek-v4.1-flash']) {
        expect(labelFor(id).name).toBe(displayModelName(id))
        expect(labelFor(id).tag).toContain(id)
      }
    })

    it('tags with the full id when colliding rows differ only in their vendor prefix', () => {
      // Kilo Code lists the same model under two prefixes; without the prefix
      // both tags would read `claude-opus-4.8` and the rows would still match.
      const ids = ['anthropic/claude-opus-4.8', 'stealth/claude-opus-4.8', 'openrouter/free', 'kilo-auto/free']
      const labelFor = modelListLabeler(ids)

      expect(new Set(ids.map(id => labelKey(labelFor(id)))).size).toBe(ids.length)
      expect(labelFor('stealth/claude-opus-4.8')).toEqual({ name: 'Opus 4.8', tag: 'stealth/claude-opus-4.8' })
    })

    it('keeps the short id as the tag when it already tells rows apart', () => {
      const labelFor = modelListLabeler(['deepseek/deepseek-flash', 'deepseek/deepseek-v4.1-flash'])

      expect(labelFor('deepseek/deepseek-flash').tag).toBe('deepseek-flash')
      expect(labelFor('deepseek/deepseek-v4.1-flash').tag).toBe('deepseek-v4.1-flash')
    })
  })

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
