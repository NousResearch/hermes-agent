import { describe, expect, it } from 'vitest'

import { applyPrintableInsert, shouldRouteMultiCharInputAsPaste } from '../components/textInput.js'

describe('applyPrintableInsert', () => {
  it('applies non-bracketed multi-character bursts immediately', () => {
    const burst = applyPrintableInsert('abc', 3, 'xxxxx')

    const repeated = [...'xxxxx'].reduce((state, ch) => applyPrintableInsert(state.value, state.cursor, ch)!, {
      cursor: 3,
      value: 'abc'
    })

    expect(burst).toEqual({ cursor: 8, value: 'abcxxxxx' })
    expect(burst).toEqual(repeated)
  })

  it('replaces the selected range for burst input', () => {
    expect(applyPrintableInsert('abZZef', 4, 'cd', { end: 4, start: 2 })).toEqual({
      cursor: 4,
      value: 'abcdef'
    })
  })

  it('rejects control or escape-bearing input', () => {
    expect(applyPrintableInsert('abc', 3, '\x1b[200~pasted')).toBeNull()
    expect(applyPrintableInsert('abc', 3, '\t')).toBeNull()
  })
})

describe('shouldRouteMultiCharInputAsPaste', () => {
  it('keeps newline-bearing chunks on the paste path', () => {
    expect(shouldRouteMultiCharInputAsPaste('hello\nworld')).toBe(true)
    expect(shouldRouteMultiCharInputAsPaste('hello\r\nworld'.replace(/\r\n/g, '\n'))).toBe(true)
  })

  it('treats repeated printable key bursts as immediate input', () => {
    expect(shouldRouteMultiCharInputAsPaste('xxxxx')).toBe(false)
  })
})
