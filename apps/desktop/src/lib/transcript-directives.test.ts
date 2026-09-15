import { describe, expect, it } from 'vitest'

import { parseTranscriptDirective } from './transcript-directives'

describe('parseTranscriptDirective', () => {
  it('parses a bare directive with no attributes', () => {
    expect(parseTranscriptDirective('::tasks')).toEqual({ name: 'tasks', attrs: {}, source: '::tasks' })
  })

  it('parses double-quoted attributes', () => {
    expect(parseTranscriptDirective('::preview{file="demo.html"}')).toEqual({
      name: 'preview',
      attrs: { file: 'demo.html' },
      source: '::preview{file="demo.html"}'
    })
  })

  it('parses multiple attributes and accepts single quotes', () => {
    expect(parseTranscriptDirective(`::vis{file="a b.html" height='480'}`)?.attrs).toEqual({
      file: 'a b.html',
      height: '480'
    })
  })

  it('lowercases attribute keys but preserves values', () => {
    expect(parseTranscriptDirective('::vis{File="A.html"}')?.attrs).toEqual({ file: 'A.html' })
  })

  it('tolerates surrounding whitespace', () => {
    expect(parseTranscriptDirective('  ::tasks{id="1"}  ')?.name).toBe('tasks')
  })

  it('rejects prose containing a directive mid-text', () => {
    expect(parseTranscriptDirective('see ::preview{file="x.html"} above')).toBeNull()
  })

  it('rejects multi-line paragraphs', () => {
    expect(parseTranscriptDirective('::preview{file="x.html"}\nmore')).toBeNull()
  })

  it('rejects C++ scope-resolution lookalikes', () => {
    expect(parseTranscriptDirective('::std')).toEqual({ name: 'std', attrs: {}, source: '::std' })
    expect(parseTranscriptDirective('std::vector<int>')).toBeNull()
    expect(parseTranscriptDirective('::Vector')).toBeNull()
  })

  it('rejects unquoted attribute values', () => {
    expect(parseTranscriptDirective('::preview{file=demo.html}')?.attrs).toEqual({})
  })

  it('bounds pathological input instead of scanning it', () => {
    expect(parseTranscriptDirective(`::x{${'a="b" '.repeat(400)}}`)).toBeNull()
  })
})
