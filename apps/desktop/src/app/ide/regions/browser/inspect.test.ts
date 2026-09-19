import { describe, expect, it } from 'vitest'

import { MAX_PICKED_HTML, parsePickedPayload, pickedElementRef } from './inspect'

describe('pickedElementRef', () => {
  it('collapses the pick into a selector-labelled element reference', () => {
    const ref = pickedElementRef({ html: '<button id="save">\n  Save\n</button>', selector: 'button#save' }, 'https://app.test/x')

    expect(ref.kind).toBe('element')
    expect(ref.label).toBe('button#save')
    expect(ref.value).toBe('button#save :: https://app.test/x :: <button id="save"> Save </button>')
  })

  it('keeps the wire value on a single line and free of backticks', () => {
    const ref = pickedElementRef(
      { html: '<code>\n`x`\n</code>', selector: 'pre > code' },
      'https://app.test'
    )

    expect(ref.value).not.toContain('\n')
    expect(ref.value).not.toContain('`')
  })

  it('truncates very large elements and long selectors', () => {
    const ref = pickedElementRef(
      { html: 'a'.repeat(MAX_PICKED_HTML + 50), selector: `div.${'x'.repeat(80)}` }
    , '')

    expect(ref.value).toContain('… (truncated)')
    expect(ref.value.length).toBeLessThan(MAX_PICKED_HTML + 200)
    expect(ref.label.length).toBeLessThanOrEqual(60)
    expect(ref.label.endsWith('…')).toBe(true)
  })

  it('falls back to a generic label for a blank selector', () => {
    const ref = pickedElementRef({ html: '<hr/>', selector: '   ' }, '')

    expect(ref.label).toBe('element')
    expect(ref.value.startsWith('element :: ')).toBe(true)
  })
})

describe('parsePickedPayload', () => {
  it('accepts the picker payload', () => {
    expect(parsePickedPayload({ html: '<a/>', selector: 'a' })).toEqual({ html: '<a/>', selector: 'a' })
  })

  it('rejects cancelled and malformed payloads', () => {
    expect(parsePickedPayload(null)).toBe(null)
    expect(parsePickedPayload(undefined)).toBe(null)
    expect(parsePickedPayload('nope')).toBe(null)
    expect(parsePickedPayload({ html: 1, selector: 'a' })).toBe(null)
    expect(parsePickedPayload({ html: '<a/>' })).toBe(null)
  })
})
