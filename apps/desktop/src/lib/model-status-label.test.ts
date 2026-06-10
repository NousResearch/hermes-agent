import { describe, expect, it } from 'vitest'

import { displayModelName, formatModelStatusLabel, reasoningEffortLabel } from './model-status-label'

describe('model-status-label', () => {
  it('formats display names consistently', () => {
    expect(displayModelName('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8')
    expect(displayModelName('openai/gpt-5.5')).toBe('GPT-5.5')
  })

  it('maps reasoning effort to compact labels', () => {
    expect(reasoningEffortLabel('high')).toBe('High')
    expect(reasoningEffortLabel('xhigh')).toBe('Max')
    expect(reasoningEffortLabel('')).toBe('')
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

  it('returns just the placeholder name when there is no model', () => {
    expect(formatModelStatusLabel('')).toBe('No model')
  })

  it('accepts dict-shaped model entries from provider config', () => {
    expect(displayModelName({ id: 'bge-m3:latest', name: 'bge-m3:latest' } as never)).toBe('Bge M3:Latest')
    expect(formatModelStatusLabel({ id: 'qwen3.7-plus', name: 'qwen3.7-plus' } as never)).toBe(
      'Qwen3.7 Plus · Med'
    )
  })
})
