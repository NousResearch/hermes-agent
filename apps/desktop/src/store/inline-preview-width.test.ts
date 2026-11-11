import { describe, expect, it } from 'vitest'

import { $inlinePreviewWide, setInlinePreviewWide } from './inline-preview-width'

describe('$inlinePreviewWide', () => {
  it('defaults to full width: an HTML reply is a document, not an image', () => {
    expect($inlinePreviewWide.get()).toBe(true)
  })

  it('round-trips a flip so the reading-column cap can be restored', () => {
    setInlinePreviewWide(false)
    expect($inlinePreviewWide.get()).toBe(false)

    setInlinePreviewWide(true)
    expect($inlinePreviewWide.get()).toBe(true)
  })
})
