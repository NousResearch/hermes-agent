import { describe, expect, it } from 'vitest'

import { shouldLetImeHandleKeyDown } from './ime-keyboard'

describe('shouldLetImeHandleKeyDown', () => {
  it('lets the IME consume Enter while native composition is active', () => {
    expect(shouldLetImeHandleKeyDown({ key: 'Enter', isComposing: true })).toBe(true)
  })

  it('lets the IME consume Enter for Process-key candidate confirmation', () => {
    expect(shouldLetImeHandleKeyDown({ key: 'Process', isComposing: false })).toBe(true)
  })

  it('lets the IME consume Enter for keyCode 229 candidate confirmation', () => {
    expect(shouldLetImeHandleKeyDown({ key: 'Enter', keyCode: 229, isComposing: false })).toBe(true)
  })

  it('lets the IME consume Enter for which 229 candidate confirmation', () => {
    expect(shouldLetImeHandleKeyDown({ key: 'Enter', which: 229, isComposing: false })).toBe(true)
  })

  it('does not treat normal Enter as IME input', () => {
    expect(shouldLetImeHandleKeyDown({ key: 'Enter', keyCode: 13, isComposing: false })).toBe(false)
  })
})
