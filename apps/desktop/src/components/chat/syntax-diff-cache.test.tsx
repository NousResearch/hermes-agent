import { cleanup, render, waitFor } from '@testing-library/react'
import type { UseShikiHighlighter } from 'react-shiki'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ highlight: vi.fn() }))
vi.mock('react-shiki', async importOriginal => {
  const actual = await importOriginal<{ useShikiHighlighter: UseShikiHighlighter }>()

  return { ...actual, useShikiHighlighter: mocks.highlight.mockImplementation(actual.useShikiHighlighter) }
})

import type { DiffLine } from './diff-lines'
import { highlightCache } from './shiki-highlight-cache'
import SyntaxDiff from './syntax-diff'

const added: DiffLine[] = [{ kind: 'add', text: 'const count = 1' }]

beforeEach(() => {
  highlightCache.clear()
  mocks.highlight.mockClear()
})
afterEach(cleanup)

it('reuses highlighted diffs across remounts while keeping incompatible content separate', async () => {
  const first = render(<SyntaxDiff language="typescript" lines={added} />)
  await waitFor(() => expect(first.container.querySelector('pre.shiki')).not.toBeNull())
  const markup = first.container.innerHTML
  first.unmount()
  mocks.highlight.mockClear()

  const reopened = render(<SyntaxDiff language="typescript" lines={added.map(line => ({ ...line }))} />)
  expect(reopened.container.innerHTML).toBe(markup)
  expect(mocks.highlight).not.toHaveBeenCalled()
  reopened.unmount()

  for (const [language, lines] of [
    ['javascript', added],
    ['typescript', [{ kind: 'remove', text: added[0].text }]],
    ['typescript', [{ kind: 'add', text: 'const count = 2' }]]
  ] satisfies [string, DiffLine[]][]) {
    mocks.highlight.mockClear()
    const changed = render(<SyntaxDiff language={language} lines={lines} />)
    expect(mocks.highlight).toHaveBeenCalled()
    await waitFor(() => expect(changed.container.querySelector('pre.shiki')).not.toBeNull())
    expect(changed.container.textContent).toBe(lines[0].text)
    const row = changed.container.querySelector('pre.shiki code > span')
    expect(row?.className).toContain(`bg-(--ui-diff-${lines[0].kind}-background)`)
    expect(row?.className).toContain('px-2.5 py-px')
    changed.unmount()
  }
})

it('keeps new text visible while highlighting settles and never caches stale output', async () => {
  const view = render(<SyntaxDiff language="typescript" lines={added} />)
  await waitFor(() => expect(view.container.querySelector('pre.shiki')).not.toBeNull())
  const next: DiffLine[] = [{ kind: 'add', text: 'const count = 2' }]
  view.rerender(<SyntaxDiff language="typescript" lines={next} />)
  expect(view.container.textContent).toBe(next[0].text)
  view.rerender(<SyntaxDiff language="typescript" lines={next} />)
  await waitFor(() => expect(view.container.querySelector('pre.shiki')?.textContent).toBe(next[0].text))
  view.unmount()
  mocks.highlight.mockClear()

  const reopened = render(<SyntaxDiff language="typescript" lines={next} />)
  expect(reopened.container.textContent).toBe(next[0].text)
  expect(mocks.highlight).not.toHaveBeenCalled()
})
