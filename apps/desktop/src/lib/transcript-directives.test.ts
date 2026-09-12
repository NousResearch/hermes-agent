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

  it('keeps braces that appear inside quoted values', () => {
    // Real failure: a follow-up prompt naming a git stash was printed as raw
    // source because `@{0}` put a brace inside the attribute value.
    const source = '::followup{p1="apply stash@{0} onto main" p2="drop stash@{2}"}'

    expect(parseTranscriptDirective(source)).toEqual({
      name: 'followup',
      attrs: { p1: 'apply stash@{0} onto main', p2: 'drop stash@{2}' },
      source
    })
  })

  it('still rejects braces outside quotes, so nesting cannot be smuggled in', () => {
    expect(parseTranscriptDirective('::preview{file="a.html" {nested}}')).toBeNull()
    expect(parseTranscriptDirective('::preview{{file="a.html"}')).toBeNull()
  })

  it('bounds pathological input instead of scanning it', () => {
    expect(parseTranscriptDirective(`::x{${'a="b" '.repeat(400)}}`)).toBeNull()
  })

  it('does not backtrack on unbalanced quotes', () => {
    // The value alternation must not turn a quote storm into exponential work.
    // Only timing is asserted — whether these shapes parse is pre-existing
    // behaviour this change deliberately leaves alone.
    const started = performance.now()

    parseTranscriptDirective(`::x{${'"'.repeat(600)}}`)
    parseTranscriptDirective(`::x{${'a="{'.repeat(200)}}`)

    expect(performance.now() - started).toBeLessThan(250)
  })
})
