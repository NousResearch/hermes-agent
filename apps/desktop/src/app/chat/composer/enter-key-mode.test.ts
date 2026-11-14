import { describe, expect, it } from 'vitest'

import { resolveComposerEnterKeyIntent } from './enter-key-mode'

describe('Desktop composer Enter key mode', () => {
  it('preserves the default Enter-to-send behavior', () => {
    expect(resolveComposerEnterKeyIntent({ enterSends: true, key: 'Enter' })).toBe('submit')
    expect(resolveComposerEnterKeyIntent({ enterSends: true, key: 'Enter', shiftKey: true })).toBe('native')
    expect(resolveComposerEnterKeyIntent({ enterSends: true, busy: true, key: 'Enter', modKey: true })).toBe('queue')
    expect(resolveComposerEnterKeyIntent({ enterSends: true, busy: false, key: 'Enter', modKey: true })).toBe('submit')
  })

  it('supports multiline-first mode: Enter inserts newline and Ctrl/Cmd+Enter sends', () => {
    expect(resolveComposerEnterKeyIntent({ enterSends: false, key: 'Enter' })).toBe('newline')
    expect(resolveComposerEnterKeyIntent({ enterSends: false, key: 'Enter', modKey: true })).toBe('submit')
    expect(resolveComposerEnterKeyIntent({ enterSends: false, busy: true, key: 'Enter', modKey: true })).toBe('queue')
  })

  it('moves steering to Shift+Enter in multiline-first mode without stealing idle newlines', () => {
    expect(resolveComposerEnterKeyIntent({ enterSends: false, canSteer: true, key: 'Enter', shiftKey: true })).toBe(
      'steer'
    )
    expect(resolveComposerEnterKeyIntent({ enterSends: false, canSteer: false, key: 'Enter', shiftKey: true })).toBe(
      'newline'
    )
  })

  it('ignores non-Enter keys', () => {
    expect(resolveComposerEnterKeyIntent({ enterSends: false, key: 'a' })).toBe('native')
  })
})
