import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { foldIncludes } from '@/lib/text'

import { HighlightMatches } from './highlight-matches'

afterEach(cleanup)

it('folds separators only for model filters and preserves the original marked text', () => {
  for (const query of ['fixture-model', 'fixture_model', 'fixture.model', 'fixture model']) {
    const text = 'Fixture Model'
    expect(foldIncludes(text, query)).toBe(true)
    const { container, unmount } = render(<HighlightMatches foldSeparators query={query} text={text} />)
    expect(container.querySelector('mark')?.textContent).toBe(text)
    expect(container.textContent).toBe(text)
    unmount()
  }

  const { container } = render(<HighlightMatches query={['fixture-model']} text="Fixture Model" />)
  expect(container.querySelector('mark')).toBeNull()
  expect(foldIncludes('Another model', 'fixture-model')).toBe(false)
})
