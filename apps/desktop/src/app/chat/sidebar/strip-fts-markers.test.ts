// Regression: the backend's session-search FTS layer wraps matched terms in
// literal '>>>' / '<<<' snippet() delimiters (hermes_state_search.py). The
// sidebar paints the snippet as plain text, so an unstripped marker renders
// rows titled ">>>foo<<<" (Aug 2026 desktop audit).
import { describe, expect, it } from 'vitest'

import { searchResultToSession, stripFtsMarkers } from './index'

describe('stripFtsMarkers', () => {
  it('strips highlight markers around the matched term', () => {
    expect(stripFtsMarkers('...replied with >>>MARCO<<< and nothing else...')).toBe(
      '...replied with MARCO and nothing else...'
    )
  })

  it('strips multiple marked terms', () => {
    expect(stripFtsMarkers('>>>alpha<<< then >>>beta<<<')).toBe('alpha then beta')
  })

  it('leaves marker-free snippets untouched', () => {
    expect(stripFtsMarkers('plain snippet text')).toBe('plain snippet text')
  })

  it('handles empty string', () => {
    expect(stripFtsMarkers('')).toBe('')
  })
})

describe('searchResultToSession', () => {
  const result = {
    model: 'test-model',
    role: null,
    session_id: '20260811_125725_5d4fac',
    session_started: 1_760_000_000,
    snippet: 'run buyer discovery',
    source: 'desktop'
  }

  it('keeps a server-result title instead of replacing it with a content snippet', () => {
    const session = searchResultToSession({ ...result, title: "Arby's Faribault, MN" })

    expect(session.title).toBe("Arby's Faribault, MN")
    expect(session.preview).toBe('run buyer discovery')
  })

  it('stays compatible with an older backend that omits title', () => {
    expect(searchResultToSession(result).title).toBeNull()
  })
})
