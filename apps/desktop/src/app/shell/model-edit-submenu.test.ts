import { describe, expect, it } from 'vitest'

import { isThinkingEnabled, normalizeEffort } from './model-edit-submenu'

describe('model-edit-submenu reasoning defaults', () => {
  it('treats empty startup state as thinking off', () => {
    expect(isThinkingEnabled('')).toBe(false)
    expect(normalizeEffort('')).toBe('')
  })

  it('keeps explicit off and effort levels distinct', () => {
    expect(isThinkingEnabled('none')).toBe(false)
    expect(normalizeEffort('none')).toBe('')
    expect(isThinkingEnabled('low')).toBe(true)
    expect(normalizeEffort('low')).toBe('low')
  })

  it('falls back unknown enabled efforts to medium', () => {
    expect(isThinkingEnabled('mystery')).toBe(true)
    expect(normalizeEffort('mystery')).toBe('medium')
  })
})
