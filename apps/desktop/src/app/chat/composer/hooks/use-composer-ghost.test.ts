import { describe, expect, it } from 'vitest'

import { firstGhostCandidate, ghostVisible } from './use-composer-ghost'

describe('firstGhostCandidate', () => {
  it('returns the first candidate text', () => {
    expect(
      firstGhostCandidate({
        candidates: [
          { kind: 'path', text: '~/Downloads/report.xlsx' },
          { kind: 'confirm', text: 'Yes, go ahead' }
        ],
        history_version: 12
      })
    ).toBe('~/Downloads/report.xlsx')
  })

  it('returns empty string for empty/malformed results', () => {
    expect(firstGhostCandidate({ candidates: [] })).toBe('')
    expect(firstGhostCandidate({})).toBe('')
    expect(firstGhostCandidate(null)).toBe('')
    expect(firstGhostCandidate(undefined)).toBe('')
    expect(firstGhostCandidate({ candidates: [{ kind: 'path' }] })).toBe('')
  })

  it('trims candidate whitespace', () => {
    expect(firstGhostCandidate({ candidates: [{ kind: 'confirm', text: '  yes  ' }] })).toBe('yes')
  })
})

describe('ghostVisible', () => {
  const base = { busy: false, dismissed: false, empty: true, ghost: 'Yes, go ahead' }

  it('shows only when idle, empty, undismissed, and a ghost exists', () => {
    expect(ghostVisible(base)).toBe(true)
  })

  it('hides while the agent is busy', () => {
    expect(ghostVisible({ ...base, busy: true })).toBe(false)
  })

  it('hides once the composer has content', () => {
    expect(ghostVisible({ ...base, empty: false })).toBe(false)
  })

  it('stays hidden after an explicit dismiss', () => {
    expect(ghostVisible({ ...base, dismissed: true })).toBe(false)
  })

  it('never shows an empty ghost', () => {
    expect(ghostVisible({ ...base, ghost: '' })).toBe(false)
  })
})
