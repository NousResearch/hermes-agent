import { describe, expect, it } from 'vitest'

import { isImeComposing } from './ime'

describe('isImeComposing', () => {
  it('detects active composition from modern keyboard events', () => {
    expect(isImeComposing({ key: 'Enter', nativeEvent: { isComposing: true } })).toBe(true)
  })

  it('detects legacy IME keyCode 229 events', () => {
    expect(isImeComposing({ key: 'Enter', nativeEvent: { keyCode: 229 } })).toBe(true)
  })

  it('detects Process key events from IME input', () => {
    expect(isImeComposing({ key: 'Process' })).toBe(true)
  })

  it('does not treat normal Enter as composition', () => {
    expect(isImeComposing({ key: 'Enter', nativeEvent: { isComposing: false, keyCode: 13 } })).toBe(false)
  })
})
