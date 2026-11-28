import { describe, expect, it } from 'vitest'

import { parsePreviewTextSelection, previewTextCommentPrompt } from './preview-text-comment'

const selection = {
  viewportWidth: 800,
  text: '  Hello\nworld 👋  ',
  url: 'http://localhost:5173/page',
  rect: { x: 12, y: 24, width: 80, height: 20 }
}

describe('preview text comments', () => {
  it('preserves exact selected text and separates page data from edit instructions', () => {
    const parsed = parsePreviewTextSelection(selection, selection.url)!
    expect(parsed.text).toBe(selection.text)
    const prompt = previewTextCommentPrompt(parsed, 'Make this warmer')
    expect(prompt).toContain(JSON.stringify(selection.text))
    expect(prompt).toContain(selection.url)
    expect(prompt).toContain('Make this warmer')
    expect(prompt).toContain('source')
  })

  it('rejects stale, remote, empty, oversized and malformed guest data', () => {
    for (const value of [
      null,
      {},
      { ...selection, text: ' ' },
      { ...selection, text: 'a'.repeat(32769) },
      { ...selection, url: 'https://example.com' },
      { ...selection, rect: { ...selection.rect, x: NaN } },
      { ...selection, rect: { ...selection.rect, width: -1 } }
    ]) {
      expect(parsePreviewTextSelection(value, selection.url)).toBeNull()
    }

    expect(parsePreviewTextSelection(selection, 'http://localhost:5173/other')).toBeNull()
  })
})
