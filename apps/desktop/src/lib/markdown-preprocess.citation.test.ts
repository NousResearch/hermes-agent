import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

// Web-citation transport markers (Gemini-style grounding) arrive in model
// output as private-use-delimited id lists. Desktop must never paint them:
// the U+E200..U+E202 glyphs render as replacement boxes ("triple bars") and
// the `turn…search…` ids are protocol noise no reader can follow to a source
// (#120587). A marker that cannot be resolved to a source is dropped, never
// invented into a link.
const P = { E200: '', E201: '', E202: '' }

describe('web citation transport markers', () => {
  it('strips a single-id marker from prose', () => {
    expect(preprocessMarkdown(`Ice floats because it is less dense than liquid water.${P.E200}citeturn0search0${P.E201}`)).toBe(
      'Ice floats because it is less dense than liquid water.'
    )
  })

  it('strips a multi-id marker list', () => {
    const input = `answer here${P.E200}citeturn0search11${P.E202}turn2search0${P.E201} and continues`
    expect(preprocessMarkdown(input)).toBe('answer here and continues')
  })

  it('strips the no-delimiter shape inside a table cell', () => {
    // The reporter's table-cell case: the marker abuts a `|` cell wall.
    const input = `| Firm${P.E200}citeturn0search1${P.E201} | US |`
    expect(preprocessMarkdown(input)).not.toContain('citeturn')
    expect(preprocessMarkdown(input)).toBe('| Firm | US |')
  })

  it('strips a bare citeturn token the model emitted without delimiters', () => {
    expect(preprocessMarkdown('See citeturn3search7 for details.')).toBe('See  for details.')
    expect(preprocessMarkdown('A citeturn0search0.')).toBe('A .')
  })

  it('strips a marker that runs to the very end of the text mid-stream', () => {
    // preprocessMarkdown runs per streaming flush; the closing U+E201 has not
    // arrived yet.
    expect(preprocessMarkdown(`Still typing${P.E200}citeturn0search0`)).toBe('Still typing')
  })

  it('fuses the words a stripped in-word marker separated, like the thinking strip does', () => {
    // The marker is transport noise, not content: deleting it inside a word
    // joins what it split (same contract as the <thinking> strip's "does not
    // fuse the words a stripped block separated" inverse — there the block is
    // replaced by a space; here the marker is INVISIBLE noise, so the visible
    // letters on either side were always adjacent).
    expect(preprocessMarkdown(`word${P.E200}citeturn0search0${P.E201}word2`)).toBe('wordword2')
  })

  it('does not touch other private-use or icon-font characters', () => {
    const icon = ''
    expect(preprocessMarkdown(`Label ${icon} stays`)).toBe(`Label ${icon} stays`)
  })

  it('keeps marker text inside inline code and math spans intact', () => {
    // The shield machinery (normalizeVisibleProse) splits on inline code and
    // math before prose rewrites; the marker rule must ride the same path.
    expect(preprocessMarkdown(`run \`x = ${P.E200}citeturn0search0${P.E201}\` please`)).toContain('citeturn')
    expect(preprocessMarkdown(`$\\sqrt[3]{8}$ and ${P.E200}citeturn0search0${P.E201}`)).toBe('$\\sqrt[3]{8}$ and ')
  })

  it('keeps stripping numeric [n] markers alongside the transport shape', () => {
    expect(preprocessMarkdown(`Both shapes[1] and ${P.E200}citeturn0search0${P.E201} go away.`)).toBe(
      'Both shapes and  go away.'
    )
  })
})
