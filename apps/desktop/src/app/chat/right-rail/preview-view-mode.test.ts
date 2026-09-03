import { describe, expect, it } from 'vitest'

import { availableViewModes, resolveViewMode } from './preview-view-mode'

// The preview rail's view-mode policy, extracted verbatim from
// LocalFilePreview: which toggles a text file offers and which one it lands
// on. Markdown renders formatted by default; an uncommitted diff always wins
// the default because reviewing changes beats reading; a user's explicit
// pick survives only while it is still offered.

describe('availableViewModes', () => {
  it('offers rendered + source for clean markdown', () => {
    expect(availableViewModes({ isMarkdown: true, hasDiff: false })).toEqual(['rendered', 'source'])
  })

  it('appends diff for changed markdown', () => {
    expect(availableViewModes({ isMarkdown: true, hasDiff: true })).toEqual(['rendered', 'source', 'diff'])
  })

  it('offers only source for clean plain text', () => {
    expect(availableViewModes({ isMarkdown: false, hasDiff: false })).toEqual(['source'])
  })

  it('offers source + diff for changed plain text', () => {
    expect(availableViewModes({ isMarkdown: false, hasDiff: true })).toEqual(['source', 'diff'])
  })
})

describe('resolveViewMode', () => {
  it('defaults clean markdown to the rendered view', () => {
    expect(resolveViewMode({ isMarkdown: true, hasDiff: false }, null)).toBe('rendered')
  })

  it('defaults changed markdown to the diff view', () => {
    expect(resolveViewMode({ isMarkdown: true, hasDiff: true }, null)).toBe('diff')
  })

  it('defaults plain text to source', () => {
    expect(resolveViewMode({ isMarkdown: false, hasDiff: false }, null)).toBe('source')
  })

  it('defaults changed plain text to the diff view', () => {
    expect(resolveViewMode({ isMarkdown: false, hasDiff: true }, null)).toBe('diff')
  })

  it('honors a user-picked mode that is still available', () => {
    expect(resolveViewMode({ isMarkdown: true, hasDiff: true }, 'source')).toBe('source')
    expect(resolveViewMode({ isMarkdown: true, hasDiff: false }, 'rendered')).toBe('rendered')
  })

  it('falls back to the auto mode when the user pick is no longer offered', () => {
    // Picked diff, then the diff disappeared (changes committed externally).
    expect(resolveViewMode({ isMarkdown: true, hasDiff: false }, 'diff')).toBe('rendered')
    // Picked rendered on a file that was never markdown.
    expect(resolveViewMode({ isMarkdown: false, hasDiff: false }, 'rendered')).toBe('source')
  })
})
