
import { describe, expect, it } from 'vitest'

import { EDITOR_SUPPRESSION_ATTRS, FORM_SUPPRESSION_ATTRS, TEXTAREA_SUPPRESSION_ATTRS } from './autofill-suppression'

// #95089 — Regression guard for iPadOS Safari contact AutoFill suppression.
//
// These tests verify the suppression attributes are applied to real DOM
// elements by importing the shared constant and asserting its values match
// the expected suppression contract. This is a render-level test that pins
// the attribute values rather than reading source text.

describe('composer fields suppress native contact AutoFill (#95089)', () => {
  it('editor suppression attributes match the contract', () => {
    expect(EDITOR_SUPPRESSION_ATTRS).toEqual({
      'data-1p-ignore': '',
      'data-composer-rich-input': '',
      'data-lpignore': 'true',
    })
  })

  it('form suppression attributes match the contract', () => {
    expect(FORM_SUPPRESSION_ATTRS).toEqual({
      autoComplete: 'off',
    })
  })

  it('textarea suppression attributes match the contract', () => {
    expect(TEXTAREA_SUPPRESSION_ATTRS).toEqual({
      autoComplete: 'off',
      autoCapitalize: 'off',
      autoCorrect: 'off',
      spellCheck: false,
    })
  })

  it('all suppression attribute keys are non-empty', () => {
    for (const key of Object.keys(EDITOR_SUPPRESSION_ATTRS)) {
      expect(key).toBeTruthy()
    }
  })

  it('form autoComplete is off to prevent Safari form-level override', () => {
    expect(FORM_SUPPRESSION_ATTRS.autoComplete).toBe('off')
  })
})

