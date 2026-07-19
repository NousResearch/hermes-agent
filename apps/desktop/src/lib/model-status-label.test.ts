import { describe, expect, it } from 'vitest'

import {
  currentPickerSelection,
  displayModelName,
  formatModelStatusLabel,
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

  it('uses i18n labels when provided', () => {
    const zhLabels: Record<string, string> = {
      minimal: '最小',
      low: '低',
      medium: '中',
      high: '高',
      xhigh: '极高',
      max: '最高',
      ultra: '超高',
      none: '关闭'
    }

    expect(reasoningEffortLabel('high', zhLabels)).toBe('高')
    expect(reasoningEffortLabel('medium', zhLabels)).toBe('中')
    expect(reasoningEffortLabel('low', zhLabels)).toBe('低')
    expect(reasoningEffortLabel('none', zhLabels)).toBe('关闭')
    expect(reasoningEffortLabel('xhigh', zhLabels)).toBe('极高')
  })

  it('falls back to built-in labels when i18n labels are missing a key', () => {
    const partialLabels: Record<string, string> = { medium: 'Mittel' }
    expect(reasoningEffortLabel('medium', partialLabels)).toBe('Mittel')
    expect(reasoningEffortLabel('high', partialLabels)).toBe('High')
  })

  it('appends fast + effort session state to the status label', () => {
    expect(formatModelStatusLabel('openai/gpt-5.5', { fastMode: true, reasoningEffort: 'high' })).toBe(
      'GPT-5.5 · Fast High'
    )
  })

  it('always surfaces the effort (default medium) so the level is visible', () => {
    expect(formatModelStatusLabel('openai/gpt-5.5', { reasoningEffort: 'medium' })).toBe('GPT-5.5 · Med')
    expect(formatModelStatusLabel('openai/gpt-5.5')).toBe('GPT-5.5 · Med')
  })

  it('localizes effort labels in status label via effortLabels', () => {
    const zhLabels: Record<string, string> = {
      low: '低',
      medium: '中',
      high: '高',
      none: '关闭'
    }

    expect(formatModelStatusLabel('openai/gpt-5.5', { reasoningEffort: 'high', effortLabels: zhLabels })).toBe(
      'GPT-5.5 · 高'
    )

    expect(formatModelStatusLabel('openai/gpt-5.5', { reasoningEffort: 'none', effortLabels: zhLabels })).toBe(
      'GPT-5.5 · 关闭'
    )

    // empty effort / default medium uses the localized fallback
    expect(formatModelStatusLabel('openai/gpt-5.5', { effortLabels: zhLabels })).toBe('GPT-5.5 · 中')
  })

  it('returns just the placeholder name when there is no model', () => {
    expect(formatModelStatusLabel('')).toBe('No model')
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
