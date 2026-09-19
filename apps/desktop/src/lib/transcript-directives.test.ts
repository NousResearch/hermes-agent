import { describe, expect, it } from 'vitest'

import { describeDirectiveParseFailure, parseTranscriptDirective } from './transcript-directives'

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

  it('rejects an attribute body that yields no attributes', () => {
    // `{file=demo.html}` used to parse into an empty-props directive, so the
    // panel mounted blank instead of reporting a drop.
    expect(parseTranscriptDirective('::preview{file=demo.html}')).toBeNull()
  })

  it('still accepts a directive with no attribute body at all', () => {
    // Rejecting a NON-EMPTY body that yields nothing must not break the
    // legitimately attribute-less forms.
    expect(parseTranscriptDirective('::sep')?.attrs).toEqual({})
    expect(parseTranscriptDirective('::sep{}')?.attrs).toEqual({})
    expect(parseTranscriptDirective('::sep{   }')?.attrs).toEqual({})
  })

  it('bounds pathological input instead of scanning it', () => {
    expect(parseTranscriptDirective(`::x{${'a="b" '.repeat(400)}}`)).toBeNull()
  })

  // Braces inside a quoted attribute value are content, not structure: a
  // follow-up prompt naming a git stash (`stash@{0}`) used to print as raw
  // source because the body rejected every brace.
  it('keeps braces that appear inside quoted values', () => {
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

  it('does not backtrack on unbalanced quotes', () => {
    // The value alternation must not turn a quote storm into exponential work.
    // Only timing is asserted — whether these shapes parse is pre-existing
    // behaviour this change deliberately leaves alone.
    const started = performance.now()

    parseTranscriptDirective(`::x{${'"'.repeat(600)}}`)
    parseTranscriptDirective(`::x{${'a="{'.repeat(200)}}`)

    expect(performance.now() - started).toBeLessThan(250)
  })

  it('reports a quoted-brace directive as parseable, not as a drop', () => {
    // The diagnostic path and the parser must agree: a directive that now
    // parses must not also be described as a failure.
    expect(describeDirectiveParseFailure('::followup{p1="stash@{0}"}')).toBeNull()
  })
})

/**
 * Tolerance for incomplete-markdown repair debris.
 *
 * A directive's attribute values are natural language, so an unpaired `*`,
 * `_`, backtick or `~~` inside a prompt makes the repair pass append a
 * synthetic closer AFTER the `}`. Strict matching turned that one stray
 * character into a silently unrendered widget, so the parse forgives inline
 * closer punctuation — and nothing else.
 */
describe('parseTranscriptDirective trailing debris', () => {
  it.each([
    ['asterisk', '*'],
    ['double asterisk', '**'],
    ['underscore', '_'],
    ['backtick', '`'],
    ['strikethrough', '~~'],
    ['mixed closers', '*_`']
  ])('parses through a trailing %s', (_label, debris) => {
    const parsed = parseTranscriptDirective(`::followup{p1="Clean the wt-* worktrees"}${debris}`)

    expect(parsed?.name).toBe('followup')
    expect(parsed?.attrs.p1).toBe('Clean the wt-* worktrees')
  })

  it('excludes the debris from source', () => {
    // Plugins echo `source` in diagnostics and fallback rendering — repair
    // debris is not part of what the model addressed.
    expect(parseTranscriptDirective('::followup{p1="a"}*')?.source).toBe('::followup{p1="a"}')
  })

  it('keeps source intact when there is no debris', () => {
    expect(parseTranscriptDirective('::followup{p1="a"}')?.source).toBe('::followup{p1="a"}')
  })

  it('tolerates debris on a bare directive', () => {
    expect(parseTranscriptDirective('::tasks*')?.name).toBe('tasks')
  })

  it('still rejects real prose after the directive', () => {
    // The tolerance must never let a directive hijack a sentence.
    expect(parseTranscriptDirective('::preview{file="x.html"} and more')).toBeNull()
    expect(parseTranscriptDirective('::preview{file="x.html"} 1')).toBeNull()
    expect(parseTranscriptDirective('::preview{file="x.html"}text')).toBeNull()
  })

  it('rejects a runaway debris tail', () => {
    expect(parseTranscriptDirective(`::tasks{a="b"}${'*'.repeat(20)}`)).toBeNull()
  })
})

/**
 * `::preview{file="…"}` with markdown punctuation in the FILENAME — the same
 * bug class as the Follow-up regression, on the directive that ships in core.
 * A glob or an underscored name is an ordinary path, and must not cost the
 * user their live preview.
 */
describe('parseTranscriptDirective preview filenames', () => {
  it.each([
    ['underscore', 'my_report.html'],
    ['leading underscore', '_draft.html'],
    ['asterisk', 'wt-*/demo.html'],
    ['backtick', 'weird`name.html'],
    ['tilde', 'backup~1.html'],
    ['brackets', 'notes[1].html'],
    ['spaces', 'my report final.html'],
    ['unicode', 'báo-cáo.html']
  ])('keeps a %s filename intact', (_label, file) => {
    const parsed = parseTranscriptDirective(`::preview{file="${file}"}`)

    expect(parsed?.name).toBe('preview')
    expect(parsed?.attrs.file).toBe(file)
  })

  it('survives repair debris caused by the filename itself', () => {
    // `wt-*/demo.html` carries the unpaired asterisk that started all this.
    const parsed = parseTranscriptDirective('::preview{file="wt-*/demo.html"}*')

    expect(parsed?.attrs.file).toBe('wt-*/demo.html')
    expect(parsed?.source).toBe('::preview{file="wt-*/demo.html"}')
  })

  it('keeps a second attribute reachable after a markdown-heavy filename', () => {
    expect(parseTranscriptDirective('::preview{file="a_b*c.html" height="480"}')?.attrs).toEqual({
      file: 'a_b*c.html',
      height: '480'
    })
  })
})
