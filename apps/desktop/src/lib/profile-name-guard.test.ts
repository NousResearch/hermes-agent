import { describe, expect, it } from 'vitest'

import { findProfileNameConflict, isOneEditApart } from './profile-name-guard'

// The behavior this pins: a typed profile name that is one keystroke from a
// profile that already exists is surfaced as a conflict BEFORE the create path
// scaffolds a stock profile + backend for the typo. The regression data is the
// reported fleet drift (veste/vestg minted beside vestr).

describe('isOneEditApart', () => {
  it.each([
    ['equal', 'vestr', 'vestr', true],
    ['substitution', 'veste', 'vestr', true],
    ['one extra character', 'vestr-frontend-devv', 'vestr-frontend-dev', true],
    ['one missing character', 'vestr-frontend-de', 'vestr-frontend-dev', true],
    ['adjacent transposition', 'defualt', 'default', true],
    ['two substitutions', 'veste', 'vastr', false],
    ['non-adjacent swap', 'aest', 'seat', false],
    ['unrelated', 'frontend-dev', 'vestr-frontend-dev', false],
    ['length gap beyond one', 'vest', 'vestr-frontend-dev', false]
  ])('%s: %s vs %s -> %s', (_label, a, b, expected) => {
    expect(isOneEditApart(a, b)).toBe(expected)
  })
})

describe('findProfileNameConflict', () => {
  const fleet = ['default', 'vestr-frontend-dev', 'vestr-curriculum-admin']

  it('flags the reported typo and names the profile that was meant', () => {
    expect(findProfileNameConflict('veste-frontend-dev', fleet)).toEqual({
      kind: 'near-miss',
      name: 'vestr-frontend-dev'
    })
    expect(findProfileNameConflict('vestg-curriculum-admin', fleet)).toEqual({
      kind: 'near-miss',
      name: 'vestr-curriculum-admin'
    })
  })

  it('treats a taken name as a duplicate, ignoring case and padding', () => {
    expect(findProfileNameConflict('  Vestr-Frontend-Dev ', fleet)).toEqual({
      kind: 'duplicate',
      name: 'vestr-frontend-dev'
    })
  })

  it('returns the same suggestion regardless of profile-list order', () => {
    expect(findProfileNameConflict('xest', ['best', 'aest'])).toEqual({ kind: 'near-miss', name: 'aest' })
    expect(findProfileNameConflict('xest', ['aest', 'best'])).toEqual({ kind: 'near-miss', name: 'aest' })
  })

  it('stays quiet for a deliberate new name, an empty one, and tiny names', () => {
    expect(findProfileNameConflict('scratch', fleet)).toBeNull()
    expect(findProfileNameConflict('   ', fleet)).toBeNull()
    // One edit from a 3-char name says nothing — too short to be a typo signal.
    expect(findProfileNameConflict('abc', ['abd'])).toBeNull()
  })
})
