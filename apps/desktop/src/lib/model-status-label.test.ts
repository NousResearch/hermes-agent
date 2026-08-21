import { describe, expect, it } from 'vitest'

import { currentPickerSelection, displayModelName, formatModelStatusLabel } from './model-status-label'
import { reasoningEffortLabel } from './reasoning-effort'

describe('model-status-label', () => {
  it('formats display names consistently', () => {
    expect(displayModelName('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8 Fast')
    expect(displayModelName('openai/gpt-5.5-fast')).toBe('GPT-5.5 Fast')
    expect(displayModelName('deepseek/deepseek-v4-pro-thinking')).toBe('Deepseek V4 Pro Thinking')
    expect(displayModelName('openai/gpt-5.5')).toBe('GPT-5.5')
    // A base model and its variant must NEVER share a display label (#88597).
    expect(displayModelName('claude-opus-5')).not.toBe(displayModelName('claude-opus-5-thinking'))
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

  it('keeps variant tags disambiguating in the status label without doubling Fast (#88597)', () => {
    // The reporter's exact case: base vs -thinking must differ in the pill.
    expect(formatModelStatusLabel('claude-opus-5')).toBe('Opus 5 · Med')
    expect(formatModelStatusLabel('claude-opus-5-thinking')).toBe('Opus 5 Thinking · Med')
    // A -fast variant already says Fast in its name; the pill must not
    // repeat it as a session-state part.
    expect(formatModelStatusLabel('openai/gpt-5.5-fast')).toBe('GPT-5.5 Fast · Med')
    // The param-driven fast flag still appends when the model id has no
    // -fast suffix.
    expect(formatModelStatusLabel('openai/gpt-5.5', { fastMode: true })).toBe('GPT-5.5 · Fast Med')
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
