import { describe, expect, it } from 'vitest'

import { historyStepAllowed } from './arrow-history-nav'

const DRAFTS = ['', '   ', 'a typed draft']

describe('historyStepAllowed', () => {
  it('refuses both arrows in every composer state when the preference is off', () => {
    for (const draft of DRAFTS) {
      for (const browsing of [false, true]) {
        expect(historyStepAllowed({ step: 'backward', browsing, draft, enabled: false })).toBe(false)
        expect(historyStepAllowed({ step: 'forward', browsing, enabled: false })).toBe(false)
      }
    }
  })

  it('refuses forward when the ring is not open', () => {
    expect(historyStepAllowed({ step: 'forward', browsing: false, enabled: true })).toBe(false)
  })

  it('allows forward only while an open ring is being walked', () => {
    expect(historyStepAllowed({ step: 'forward', browsing: true, enabled: true })).toBe(true)
  })

  it('opens backward from an untouched composer', () => {
    for (const draft of ['', '   ']) {
      expect(historyStepAllowed({ step: 'backward', browsing: false, draft, enabled: true })).toBe(true)
    }
  })

  it('refuses backward over a typed draft the user did not recall', () => {
    expect(
      historyStepAllowed({ step: 'backward', browsing: false, draft: 'a typed draft', enabled: true })
    ).toBe(false)
  })

  it('keeps stepping backward while browsing, whatever the composer holds', () => {
    for (const draft of DRAFTS) {
      expect(historyStepAllowed({ step: 'backward', browsing: true, draft, enabled: true })).toBe(true)
    }
  })
})
