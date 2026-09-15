
import { describe, expect, it } from 'vitest'

import {
  EDITOR_SUPPRESSION_ATTRS,
  FORM_SUPPRESSION_ATTRS,
  TEXTAREA_SUPPRESSION_ATTRS
} from './autofill-suppression'

// #95089 — iPadOS Safari hardware-keyboard contact AutoFill suppression.
//
// These tests verify the suppression attribute constants match the expected
// contract. The actual DOM application is tested in the component-level
// tests via the shared constant. This file is pure (no jsdom needed).

describe('rich composer AutoFill suppression (#95089)', () => {
  it('editor suppression attributes match the contract', () => {
    expect(EDITOR_SUPPRESSION_ATTRS).toEqual({
      'data-1p-ignore': '',
      'data-composer-rich-input': '',
      'data-lpignore': 'true'
    })
  })

  it('form suppression attributes match the contract', () => {
    expect(FORM_SUPPRESSION_ATTRS).toEqual({ autoComplete: 'off' })
  })

  it('textarea suppression attributes match the contract', () => {
    expect(TEXTAREA_SUPPRESSION_ATTRS).toEqual({
      autoComplete: 'off',
      autoCapitalize: 'off',
      autoCorrect: 'off',
      spellCheck: false
    })
  })

  it('editor attributes include all required suppression markers', () => {
    const keys = Object.keys(EDITOR_SUPPRESSION_ATTRS)
    expect(keys).toContain('data-1p-ignore')
    expect(keys).toContain('data-composer-rich-input')
    expect(keys).toContain('data-lpignore')
  })

  it('form autoComplete is off to prevent Safari form-level override', () => {
    expect(FORM_SUPPRESSION_ATTRS.autoComplete).toBe('off')
  })

  it('textarea autoComplete is off', () => {
    expect(TEXTAREA_SUPPRESSION_ATTRS.autoComplete).toBe('off')
  })
})

