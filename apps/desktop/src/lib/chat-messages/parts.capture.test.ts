// Regression for #84361: the bare-word capture branch absorbed trailing
// sentence punctuation (`open MEDIA:/tmp/a.pdf.` captured `/tmp/a.pdf.`) and
// accepted degenerate captures (a lone quote, `...`, a bare word with neither
// a path separator nor a file extension), rendering dead file links.
import { describe, expect, it } from 'vitest'

import { mediaTagValues, renderMediaTags } from './parts'

const card = (path: string) => `[File: ${path.split(/[/\\]/).pop()}](#media:${encodeURIComponent(path)})`

describe('MEDIA tag capture validity', () => {
  it('does not absorb trailing sentence punctuation from a bare capture', () => {
    expect(renderMediaTags('open MEDIA:/tmp/a.pdf.')).toBe(`open ${card('/tmp/a.pdf')}.`)
    expect(renderMediaTags('MEDIA:/tmp/report.pdf!')).toBe(`${card('/tmp/report.pdf')}!`)
    expect(mediaTagValues('open MEDIA:/tmp/a.pdf.')).toEqual(['/tmp/a.pdf'])
  })

  it('keeps punctuation that is INSIDE a quoted capture', () => {
    // mediaTagValues returns raw values quotes-intact; renderMediaTags strips.
    expect(mediaTagValues("MEDIA:'/tmp/stop!.md'")).toEqual(["'/tmp/stop!.md'"])
    expect(renderMediaTags("MEDIA:'/tmp/stop!.md'")).toBe(card('/tmp/stop!.md'))
    expect(renderMediaTags('MEDIA:"/tmp/a b.md." x')).toBe(`${card('/tmp/a b.md.')} x`)
  })

  it('leaves degenerate captures as plain text instead of a dead link', () => {
    // A lone apostrophe, ellipsis, or bare word: neither a path separator nor
    // a file extension — not a deliverable the app can open.
    expect(mediaTagValues('MEDIA:...')).toEqual([])
    expect(mediaTagValues("MEDIA:'")).toEqual([])
    expect(mediaTagValues('MEDIA:download')).toEqual([])
    expect(renderMediaTags('MEDIA:...')).toBe('MEDIA:...')
    expect(renderMediaTags('MEDIA:download')).toBe('MEDIA:download')
  })

  it('still links relative paths with a file extension', () => {
    expect(mediaTagValues('MEDIA:report.md prose')).toEqual(['report.md'])
    expect(renderMediaTags('MEDIA:report.md prose')).toBe(`${card('report.md')} prose`)
  })
})

describe('MEDIA-GALLERY block collapsing', () => {
  it('collapses a block of images into a single #gallery link', () => {
    const gallery = [
      'MEDIA-GALLERY: Login flow (1200ms)',
      'MEDIA:/tmp/login-1.png',
      'MEDIA:/tmp/login-2.png',
      'MEDIA:/tmp/login-3.png',
      '<!-- /MEDIA-GALLERY -->'
    ].join('\n')

    const rendered = renderMediaTags(gallery)

    expect(rendered.startsWith('[Gallery: 3 images](#gallery:')).toBe(true)
    // Non-image MEDIA in/around the block still renders normally.
    expect(renderMediaTags(`${gallery}\nMEDIA:/tmp/clip.mp4`)).toContain('[Video: clip.mp4](#media:%2Ftmp%2Fclip.mp4)')
  })

  it('leaves the block alone when there are fewer than two gallery images', () => {
    const gallery = ['MEDIA-GALLERY: Lonely', 'MEDIA:/tmp/only.png', '<!-- /MEDIA-GALLERY -->'].join('\n')

    const rendered = renderMediaTags(gallery)
    // Not collapsed to a gallery; the single image still renders as a thumbnail
    // and the gallery markers are preserved.
    expect(rendered).toContain('MEDIA-GALLERY: Lonely')
    expect(rendered).toContain('[Image: only.png](#media:%2Ftmp%2Fonly.png)')
    expect(rendered).not.toContain('#gallery:')
  })

  it('ignores unterminated MEDIA-GALLERY blocks', () => {
    const gallery = 'MEDIA-GALLERY: No close\nMEDIA:/tmp/a.png\nMEDIA:/tmp/b.png'

    expect(renderMediaTags(gallery)).toContain('MEDIA-GALLERY: No close')
  })
})
