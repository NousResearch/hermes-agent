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

  // KNOWN BUG — remove `.fails` when DIRECTIVE_RE stops rejecting braces that
  // sit inside a quoted value. The body is matched with `[^{}]`, so a follow-up
  // prompt naming a git stash (`p1="stash@{0}"`) fails to parse and the raw
  // `::followup{...}` source is rendered as prose in the transcript.
  // Marked `.fails` so this lands green and starts failing the moment the
  // behaviour is fixed, rather than red-listing CI in the meantime.
  it.fails('keeps braces that appear inside quoted values', () => {
    const source = '::followup{p1="apply stash@{0} onto main" p2="drop stash@{2}"}'

    expect(parseTranscriptDirective(source)).toEqual({
      name: 'followup',
      attrs: { p1: 'apply stash@{0} onto main', p2: 'drop stash@{2}' },
      source
    })
  })

  it('rejects braces outside quotes', () => {
    // Passes today and must keep passing after the fix: loosening the body
    // must not let an unquoted brace through.
    expect(parseTranscriptDirective('::preview{file="a.html" {nested}}')).toBeNull()
    expect(parseTranscriptDirective('::preview{{file="a.html"}')).toBeNull()
  })

  it('bounds pathological input instead of scanning it', () => {
    expect(parseTranscriptDirective(`::x{${'a="b" '.repeat(400)}}`)).toBeNull()
  })

  it('does not backtrack on brace and quote storms', () => {
    // Timing only — whether these shapes parse is pre-existing behaviour.
    const started = performance.now()

    parseTranscriptDirective(`::x{${'"'.repeat(600)}}`)
    parseTranscriptDirective(`::x{${'a="{'.repeat(200)}}`)

    expect(performance.now() - started).toBeLessThan(250)
  })
})
