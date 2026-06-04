import { describe, expect, it } from 'vitest'

import { isImeComposing } from './ime'

describe('isImeComposing', () => {
  it('detects active IME composition on native keyboard events', () => {
    expect(isImeComposing({ nativeEvent: { isComposing: true } })).toBe(true)
  })

  it('detects legacy process-key IME events by keyCode 229', () => {
    expect(isImeComposing({ nativeEvent: { keyCode: 229 } })).toBe(true)
  })

  it('does not treat ordinary Enter events as composition', () => {
    expect(isImeComposing({ nativeEvent: { isComposing: false, keyCode: 13 } })).toBe(false)
  })
})
