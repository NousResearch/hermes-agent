import { describe, expect, it } from 'vitest'

import {
  currentPickerSelection,
  displayModelName,
  formatModelPillLabel,
  formatReasoningPillLabel,
  isThinkingEnabled,
  normalizeReasoningEffort,
  REASONING_EFFORT_OPTIONS,
  reasoningEffortLabel
} from './model-status-label'

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

  it('maps reasoning effort to compact labels', () => {
    expect(reasoningEffortLabel('high')).toBe('High')
    expect(reasoningEffortLabel('xhigh')).toBe('XHigh')
    expect(reasoningEffortLabel('max')).toBe('Max')
    expect(reasoningEffortLabel('ultra')).toBe('Ultra')
    expect(reasoningEffortLabel('')).toBe('')
  })

  it('exposes the full effort options list for the picker radio group', () => {
    // All 7 reasoning levels — preserving `max` + `ultra` (added to
    // VALID_REASONING_EFFORTS via PR #62650) so existing user presets for
    // those levels do not silently fall back to `medium` (see F1 in the
    // PR #51524 review).
    expect(REASONING_EFFORT_OPTIONS.length).toBe(7)
    expect(REASONING_EFFORT_OPTIONS.map(option => option.value)).toEqual([
      'minimal',
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
      'ultra'
    ])
    // Lock the labelKey for `xhigh` to `'xhigh'` rather than `'max'` — the
    // dropdown label and the pill's compact label use different keys
    // (`copy.xhigh` for the full label, `REASONING_LABELS.xhigh = 'XHigh'`
    // for the compact pill). Pointing `xhigh` at `'max'` here would render
    // the dropdown as 'Max' while the pill renders as 'XHigh' — a UX bug.
    expect(REASONING_EFFORT_OPTIONS.find(option => option.value === 'xhigh')).toMatchObject({ labelKey: 'xhigh' })
    expect(REASONING_EFFORT_OPTIONS.find(option => option.value === 'max')).toMatchObject({ labelKey: 'max' })
    expect(REASONING_EFFORT_OPTIONS.find(option => option.value === 'ultra')).toMatchObject({ labelKey: 'ultra' })
  })

  it('treats empty effort as thinking-on (Hermes default) and only none as off', () => {
    expect(isThinkingEnabled('')).toBe(true)
    expect(isThinkingEnabled('medium')).toBe(true)
    expect(isThinkingEnabled('none')).toBe(false)
  })

  it('normalizes effort to a valid radio value, with "none" → "" and unknown → medium', () => {
    expect(normalizeReasoningEffort('high')).toBe('high')
    expect(normalizeReasoningEffort('none')).toBe('')
    // Regression: `max` and `ultra` must round-trip instead of falling back
    // to `medium`. Before PR #62650 the helper only had 5 levels; adding
    // levels without updating this list silently demoted user presets.
    expect(normalizeReasoningEffort('max')).toBe('max')
    expect(normalizeReasoningEffort('ultra')).toBe('ultra')
    expect(normalizeReasoningEffort('gibberish')).toBe('medium')
    expect(normalizeReasoningEffort('')).toBe('medium')
  })

  it('formats the model pill as just the name (no effort suffix)', () => {
    expect(formatModelPillLabel('openai/gpt-5.5', { fastMode: true })).toBe('GPT-5.5 · Fast')
    expect(formatModelPillLabel('openai/gpt-5.5')).toBe('GPT-5.5')
  })

  it('appends · Fast to the model pill when the active variant is a `-fast` sibling', () => {
    expect(formatModelPillLabel('openai/gpt-5.5-fast')).toBe('GPT-5.5 · Fast')
  })

  it('returns just the placeholder name when the model is empty', () => {
    expect(formatModelPillLabel('')).toBe('No model')
    expect(formatModelPillLabel('   ')).toBe('No model')
  })

  it('formats the reasoning pill label, falling back to Med for empty effort', () => {
    expect(formatReasoningPillLabel('high')).toBe('High')
    // `xhigh`'s compact label is `'XHigh'` (its own slot in REASONING_LABELS)
    // — not `'Max'`, which is reserved for the explicit `max` level.
    expect(formatReasoningPillLabel('xhigh')).toBe('XHigh')
    expect(formatReasoningPillLabel('max')).toBe('Max')
    expect(formatReasoningPillLabel('ultra')).toBe('Ultra')
    expect(formatReasoningPillLabel('none')).toBe('Off')
    expect(formatReasoningPillLabel('')).toBe('Med')
  })

  describe('currentPickerSelection', () => {
    const store = { model: 'opus', provider: 'anthropic' }
    const options = { model: 'hermes-4', provider: 'nous' }

    it('prefers the sticky composer pick over the profile default pre-session', () => {
      expect(currentPickerSelection(false, store, options)).toEqual(store)
    })

    it('lets the live session model.options win when a session exists', () => {
      expect(currentPickerSelection(true, store, options)).toEqual(options)
    })

    it('falls back to options when the store is empty', () => {
      expect(currentPickerSelection(false, { model: '', provider: '' }, options)).toEqual(options)
    })

    it('falls back to the store while options are still loading', () => {
      expect(currentPickerSelection(true, store, undefined)).toEqual(store)
    })
  })
})
