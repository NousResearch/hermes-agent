// Regression for #96657: an unquoted MEDIA: path with interior spaces rendered
// a card for the text before the first space and left the rest as prose.
import { describe, expect, it } from 'vitest'

import { appendAssistantTextPart, assistantTextPart, chatMessageText, mediaTagValues, renderMediaTags } from './parts'

const SPACED = '/home/hermes/Morten - Nobly Kickoff - Opening and cue cards EN.docx'
const CARD = `[File: Morten - Nobly Kickoff - Opening and cue cards EN.docx](#media:${encodeURIComponent(SPACED)})`

describe('renderMediaTags with interior spaces', () => {
  it('keeps the whole spaced path in one card on every surface that reads MEDIA tags', () => {
    expect(renderMediaTags(`MEDIA:${SPACED}`)).toBe(CARD)
    expect(renderMediaTags(`Here you go: MEDIA:${SPACED} — enjoy`)).toBe(`Here you go: ${CARD} — enjoy`)
    expect(renderMediaTags('MEDIA:C:\\Users\\Morten\\My Report.docx')).toBe(
      '[File: My Report.docx](#media:C%3A%5CUsers%5CMorten%5CMy%20Report.docx)'
    )
    expect(mediaTagValues(`ready\nMEDIA:${SPACED}\nMEDIA:/tmp/a.png`)).toEqual([SPACED, '/tmp/a.png'])
  })

  it('settles on the complete path when the stream splits inside it', () => {
    const chunks = ['ready\nMEDIA:/tmp/AI', ' Brain/re', 'port.pdf', '\nall done']
    let parts = appendAssistantTextPart([], chunks[0])

    // Mid-stream the truncated prefix may render as a card; the next delta must undo it.
    for (const chunk of chunks.slice(1)) {
      parts = appendAssistantTextPart(parts, chunk)
    }

    expect(chatMessageText({ id: 'a', parts, role: 'assistant' })).toBe(
      'ready\n[File: report.pdf](#media:%2Ftmp%2FAI%20Brain%2Freport.pdf)\nall done'
    )
  })
})

describe('inline-code MEDIA paths', () => {
  const card = (path: string) => `[File: ${path.split(/[/\\]/).pop()}](#media:${encodeURIComponent(path)})`

  it('does not swallow a trailing backtick on relative or unknown-extension paths', () => {
    expect(mediaTagValues('MEDIA:report.md` prose')).toEqual(['report.md'])
    expect(mediaTagValues('MEDIA:/tmp/file.unknown` prose')).toEqual(['/tmp/file.unknown'])
    expect(mediaTagValues('`MEDIA:notes.log` prose')).toEqual(['notes.log'])
    expect(mediaTagValues('MEDIA:draft.md`，打开复制')).toEqual(['draft.md'])
    expect(renderMediaTags('MEDIA:report.md` prose')).toBe(`${card('report.md')} prose`)
    expect(renderMediaTags('MEDIA:/tmp/file.unknown`，打开')).toBe(`${card('/tmp/file.unknown')}，打开`)
    expect(renderMediaTags('MEDIA:report.md`')).toBe(card('report.md'))
    expect(renderMediaTags('`MEDIA:/tmp/file.unknown`')).toBe(card('/tmp/file.unknown'))
    expect(mediaTagValues("MEDIA:/tmp/john's.unknown x")).toEqual(["/tmp/john's.unknown"])
    expect(renderMediaTags('MEDIA:"/tmp/a b.md" x')).toBe(`${card('/tmp/a b.md')} x`)
    expect(mediaTagValues('MEDIA:/tmp/file.unknown" prose')).toEqual(['/tmp/file.unknown'])
  })

  it('keeps absolute markdown backtick wraps and leaves the anchored branch intact', () => {
    expect(mediaTagValues('MEDIA:/Users/a/report.md` followed by prose')).toEqual(['/Users/a/report.md'])
    expect(mediaTagValues('MEDIA:/Users/a/draft.md`，打开复制')).toEqual(['/Users/a/draft.md'])
    expect(mediaTagValues('`MEDIA:/Users/a/report.md` prose')).toEqual(['/Users/a/report.md'])
    expect(renderMediaTags('`MEDIA:/dir with space/f.md`')).toBe(card('/dir with space/f.md'))
    expect(mediaTagValues("MEDIA:/tmp/john's.md x")).toEqual(["/tmp/john's.md"])
    expect(renderMediaTags("MEDIA:'/tmp/a b.md' x")).toBe(`${card('/tmp/a b.md')} x`)
    expect(renderMediaTags('MEDIA:/tmp/a.png')).toBe('[Image: a.png](#media:%2Ftmp%2Fa.png)')
  })
})

describe('DSML tool-call leakage', () => {
  it('hides every non-empty opening-tag prefix while streaming', () => {
    const openingTag = '<｜DSML｜tool_calls>'

    for (let split = 1; split < openingTag.length; split += 1) {
      const parts = appendAssistantTextPart([], `Before ${openingTag.slice(0, split)}`)

      expect(chatMessageText({ id: `assistant-${split}`, parts, role: 'assistant' })).toBe('Before ')
    }
  })

  it('hides a tool-call block when its tags arrive in separate stream deltas', () => {
    let parts = appendAssistantTextPart([], 'Before <｜DSML｜tool_')

    expect(chatMessageText({ id: 'assistant', parts, role: 'assistant' })).toBe('Before ')

    parts = appendAssistantTextPart(parts, 'calls><｜DSML｜invoke name="terminal">pwd</｜DSML｜invoke>')

    expect(chatMessageText({ id: 'assistant', parts, role: 'assistant' })).toBe('Before ')

    parts = appendAssistantTextPart(parts, '</｜DSML｜tool_calls> After')

    expect(chatMessageText({ id: 'assistant', parts, role: 'assistant' })).toBe('Before  After')
  })

  it('restores an opening-tag prefix when later text proves it was ordinary prose', () => {
    let parts = appendAssistantTextPart([], 'Before <｜DSML｜tool_')
    parts = appendAssistantTextPart(parts, 'tips are useful.')

    expect(chatMessageText({ id: 'assistant', parts, role: 'assistant' })).toBe(
      'Before <｜DSML｜tool_tips are useful.'
    )
  })

  it('settles an unfinished opening-tag prefix as ordinary text in a final response', () => {
    const text = 'Before <｜DSML｜tool_'

    expect(assistantTextPart(text)).toMatchObject({ type: 'text', text })
  })

  it('hides a complete DSML tool-call block from final assistant content', () => {
    const part = assistantTextPart('Before <｜DSML｜tool_calls><｜DSML｜invoke name="terminal">pwd</｜DSML｜invoke></｜DSML｜tool_calls> After')

    expect(part).toMatchObject({ type: 'text', text: 'Before  After' })
  })

  it('preserves ordinary text with similar ASCII characters', () => {
    const text = 'Use <|DSML|tool_calls> as a literal example, not a tool call.'

    expect(assistantTextPart(text)).toMatchObject({ type: 'text', text })
  })
})
