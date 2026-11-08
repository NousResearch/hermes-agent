import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { HighlightMatches } from './highlight-matches'

const marksOf = (container: HTMLElement) => Array.from(container.querySelectorAll('mark')).map(m => m.textContent)

describe('HighlightMatches', () => {
  it('wraps every case-insensitive occurrence in a <mark> without altering the text', () => {
    const { container } = render(<HighlightMatches query="ro" text="Grok 4.5 Retro" />)

    expect(container.textContent).toBe('Grok 4.5 Retro')
    expect(marksOf(container)).toEqual(['ro', 'ro'])
  })

  it('renders plain text when the query is empty or does not occur in this label', () => {
    // Non-occurrence matters: rows can match the filter on id/slug while the
    // display label contains no occurrence — must not crash or mis-mark.
    for (const query of ['', '   ', 'zzz']) {
      const { container } = render(<HighlightMatches query={query} text="Fable 5" />)

      expect(container.textContent).toBe('Fable 5')
      expect(container.querySelector('mark')).toBeNull()
    }
  })

  it('normalizes the query the same way the pickers filter (trim + lowercase)', () => {
    const { container } = render(<HighlightMatches query="  GROK " text="grok-4.5" />)

    expect(marksOf(container)).toEqual(['grok'])
  })

  it('marks every term of a multi-term query (palette AND semantics)', () => {
    const { container } = render(<HighlightMatches query={['open', 'set']} text="Open Settings" />)

    expect(container.textContent).toBe('Open Settings')
    expect(marksOf(container)).toEqual(['Open', 'Set'])
  })

  it('merges overlapping and adjacent term ranges into one mark', () => {
    const { container } = render(<HighlightMatches query={['gro', 'rok']} text="Grok" />)

    expect(marksOf(container)).toEqual(['Grok'])
  })
})
