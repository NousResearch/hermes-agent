import { describe, expect, it } from 'vitest'

import { segmentTranscriptDirectives } from './transcript-directives'

const directive = (text: string) => segmentTranscriptDirectives(text)?.find(segment => segment.kind === 'directive')

describe('segmentTranscriptDirectives', () => {
  it('parses a bare directive with no attributes', () => {
    expect(segmentTranscriptDirectives('::tasks')).toEqual([
      { kind: 'directive', directive: { name: 'tasks', attrs: {}, source: '::tasks' } }
    ])
  })

  it('parses multiple attributes and accepts single quotes', () => {
    expect(directive(`::vis{file="a b.html" height='480'}`)).toEqual({
      kind: 'directive',
      directive: {
        name: 'vis',
        attrs: { file: 'a b.html', height: '480' },
        source: `::vis{file="a b.html" height='480'}`
      }
    })
  })

  it('lowercases attribute keys but preserves values', () => {
    expect(directive('::vis{File="A.html"}')).toMatchObject({ directive: { attrs: { file: 'A.html' } } })
  })

  it('ignores unquoted attribute values', () => {
    expect(directive('::preview{file=demo.html}')).toMatchObject({ directive: { attrs: {} } })
  })

  it('does not read C++ scope resolution or capitalised names as directives', () => {
    expect(segmentTranscriptDirectives('std::vector<int>')).toBeNull()
    expect(segmentTranscriptDirectives('::Vector')).toBeNull()
  })

  it('leaves a directive whose attributes do not parse as text', () => {
    expect(segmentTranscriptDirectives('::x{unclosed')).toBeNull()
  })

  it('bounds pathological input instead of scanning it', () => {
    expect(segmentTranscriptDirectives(`::x{${'a="b" '.repeat(1000)}}`)).toBeNull()
  })
})

describe('segmentTranscriptDirectives / long attribute values', () => {
  it('parses a directive carrying two signed URLs', () => {
    // Signed CDN URLs run 500-1,000 chars; a model card names a model and its thumbnail.
    const signed = (name: string) => `https://cdn.example.com/${name}?Signature=${'s'.repeat(800)}&Key-Pair-Id=K`
    const text = `::model{url="${signed('m.glb')}" thumb="${signed('t.png')}" title="Fox"}`

    expect(directive(text)).toMatchObject({
      directive: { attrs: { url: signed('m.glb'), thumb: signed('t.png'), title: 'Fox' } }
    })
  })
})
