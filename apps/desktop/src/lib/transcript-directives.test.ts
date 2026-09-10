import { describe, expect, it } from 'vitest'

import { isDirectiveInProgress, segmentTranscriptDirectives } from './transcript-directives'

/** The directives found in a paragraph, prose discarded. */
const directivesIn = (text: string) =>
  (segmentTranscriptDirectives(text) ?? []).flatMap(segment => (segment.kind === 'directive' ? [segment.directive] : []))

const kindsIn = (text: string) => segmentTranscriptDirectives(text)?.map(segment => segment.kind)

describe('segmentTranscriptDirectives', () => {
  it('parses a bare directive with no attributes', () => {
    expect(directivesIn('::tasks')).toEqual([{ name: 'tasks', attrs: {}, source: '::tasks' }])
  })

  it('parses double-quoted attributes', () => {
    expect(directivesIn('::preview{file="demo.html"}')).toEqual([
      { name: 'preview', attrs: { file: 'demo.html' }, source: '::preview{file="demo.html"}' }
    ])
  })

  it('parses multiple attributes and accepts single quotes', () => {
    expect(directivesIn(`::vis{file="a b.html" height='480'}`)[0].attrs).toEqual({ file: 'a b.html', height: '480' })
  })

  it('lowercases attribute keys but preserves values', () => {
    expect(directivesIn('::vis{File="A.html"}')[0].attrs).toEqual({ file: 'A.html' })
  })

  it('rejects unquoted attribute values', () => {
    expect(directivesIn('::preview{file=demo.html}')[0].attrs).toEqual({})
  })

  it('parses two directives merged onto one line (the model-slop case)', () => {
    const parsed = directivesIn('::onboarding{step="name" value="karan"} ::onboarding{step="focus"}')

    expect(parsed.map(p => p.attrs)).toEqual([{ step: 'name', value: 'karan' }, { step: 'focus' }])
  })

  it('parses directives split across lines in one paragraph', () => {
    expect(directivesIn('::onboarding{step="look"}\n::onboarding{step="layout"}').map(p => p.attrs.step)).toEqual([
      'look',
      'layout'
    ])
  })

  // The failure this exists for: the card is swallowed AND the markup is shown,
  // which on a step whose card is the only way forward ends the conversation.
  it('recovers a directive the model tacked onto the end of a sentence', () => {
    const segments = segmentTranscriptDirectives('Pick a color you like. ::onboarding{step="look"}')

    expect(segments?.map(s => (s.kind === 'prose' ? s.text.trim() : s.directive.attrs.step))).toEqual([
      'Pick a color you like.',
      'look'
    ])
  })

  it('keeps the order of prose and directives, wherever they fall', () => {
    expect(kindsIn('::onboarding{step="name" value="bk"} Good to meet you.')).toEqual(['directive', 'prose'])
    expect(kindsIn('see ::preview{file="x.html"} above')).toEqual(['prose', 'directive', 'prose'])
  })

  it('leaves paragraphs with no directive alone', () => {
    expect(segmentTranscriptDirectives('just some text')).toBeNull()
    expect(segmentTranscriptDirectives('a ratio of 3::1')).toBeNull()
  })

  // A directive starts a word or it is not one — otherwise every C++ snippet in
  // the transcript is a directive, and a name has to be lowercase to be one.
  it('never reads a scope-resolution operator as a directive', () => {
    expect(segmentTranscriptDirectives('call std::vector::push_back here')).toBeNull()
    expect(segmentTranscriptDirectives('::Vector')).toBeNull()
  })

  it('bounds pathological input instead of scanning it', () => {
    expect(segmentTranscriptDirectives(`::x{${'a="b" '.repeat(400)}}`)).toBeNull()
  })

  // Half a directive is not one — a card whose attributes were silently
  // dropped is broken in a way nobody can see from the outside.
  it('leaves a directive whose attributes it could not read as text', () => {
    expect(segmentTranscriptDirectives('::preview{file="unclosed.html"')).toBeNull()
  })
})

describe('isDirectiveInProgress', () => {
  it('flags a partially-streamed directive paragraph (the flash window)', () => {
    // The real symptom: ~3-char deltas leave the directive unparseable until
    // the closing `}` lands. Every prefix along the way must be withheld.
    const full = '::ask{question="What sounds better?" options="I have something in mind|Automate"}'

    for (let end = 2; end < full.length; end += 3) {
      expect(isDirectiveInProgress(full.slice(0, end))).toBe(true)
    }
  })

  it('flags the lone-colon prefix one delta earlier', () => {
    expect(isDirectiveInProgress(':')).toBe(true)
  })

  it('flags a complete directive line too (callers gate on streaming, not shape)', () => {
    expect(isDirectiveInProgress('::onboarding{step="connectors"}')).toBe(true)
    expect(isDirectiveInProgress('  ::tasks')).toBe(true)
  })

  it('leaves ordinary streaming prose alone', () => {
    expect(isDirectiveInProgress('just some text')).toBe(false)
    expect(isDirectiveInProgress('a colon: mid-sentence')).toBe(false)
    expect(isDirectiveInProgress('std::vector<int>')).toBe(false)
    expect(isDirectiveInProgress(':single-colon-emoji-ish')).toBe(false)
    expect(isDirectiveInProgress('')).toBe(false)
  })
})
