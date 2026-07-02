import { describe, expect, it } from 'vitest'

import { defaultPreviewViewMode } from './preview-file'

describe('defaultPreviewViewMode', () => {
  it('keeps Markdown previews rendered even when the file has a diff', () => {
    expect(defaultPreviewViewMode(true, true)).toBe('rendered')
  })

  it('defaults changed non-Markdown text files to diff review', () => {
    expect(defaultPreviewViewMode(false, true)).toBe('diff')
  })

  it('defaults clean non-Markdown text files to source', () => {
    expect(defaultPreviewViewMode(false, false)).toBe('source')
  })
})
