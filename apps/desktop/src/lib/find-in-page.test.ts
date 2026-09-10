import { describe, expect, it } from 'vitest'
import { formatMatchLabel, findBarKeyAction, findBarClaimsCombo } from '@/lib/find-in-page'

describe('formatMatchLabel', () => {
  it('returns empty for no query', () => {
    expect(formatMatchLabel('', 0, 0)).toBe('')
  })

  it('returns 0/0 for query with no matches', () => {
    expect(formatMatchLabel('test', 0, 0)).toBe('0/0')
  })

  it('formats match count', () => {
    expect(formatMatchLabel('test', 3, 12)).toBe('3/12')
  })

  it('clamps ordinal to count', () => {
    expect(formatMatchLabel('test', 20, 5)).toBe('5/5')
  })

  it('clamps negative ordinal to 0', () => {
    expect(formatMatchLabel('test', -1, 5)).toBe('0/5')
  })

  it('handles NaN ordinal', () => {
    expect(formatMatchLabel('test', NaN, 5)).toBe('0/5')
  })
})

describe('findBarKeyAction', () => {
  it('Escape closes', () => {
    expect(findBarKeyAction({ key: 'Escape' })).toBe('close')
  })

  it('Ctrl+G goes next', () => {
    expect(findBarKeyAction({ key: 'g', ctrlKey: true })).toBe('next')
  })

  it('Ctrl+Shift+G goes previous', () => {
    expect(findBarKeyAction({ key: 'G', ctrlKey: true, shiftKey: true })).toBe('previous')
  })

  it('Enter in input goes next', () => {
    expect(findBarKeyAction({ key: 'Enter' }, { inInput: true })).toBe('next')
  })

  it('Enter outside input returns null', () => {
    expect(findBarKeyAction({ key: 'Enter' })).toBeNull()
  })

  it('Alt disqualifies all', () => {
    expect(findBarKeyAction({ key: 'Escape', altKey: true })).toBeNull()
    expect(findBarKeyAction({ key: 'g', ctrlKey: true, altKey: true })).toBeNull()
  })

  it('regular typing returns null', () => {
    expect(findBarKeyAction({ key: 'a' })).toBeNull()
  })
})

describe('findBarClaimsCombo', () => {
  it('claims mod+g', () => {
    expect(findBarClaimsCombo('mod+g')).toBe(true)
  })

  it('claims mod+shift+g', () => {
    expect(findBarClaimsCombo('mod+shift+g')).toBe(true)
  })

  it('claims escape', () => {
    expect(findBarClaimsCombo('escape')).toBe(true)
  })

  it('does not claim other combos', () => {
    expect(findBarClaimsCombo('mod+k')).toBe(false)
    expect(findBarClaimsCombo('ctrl+r')).toBe(false)
  })
})
