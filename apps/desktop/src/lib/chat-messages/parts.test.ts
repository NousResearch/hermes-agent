// Regression for #96657: an unquoted MEDIA: path with interior spaces rendered
// a card for the text before the first space and left the rest as prose.
import { describe, expect, it } from 'vitest'

import { appendAssistantTextPart, chatMessageText, mediaTagValues, renderMediaTags } from './parts'

const SPACED = '/home/hermes/Morten - Nobly Kickoff - Opening and cue cards EN.docx'
const CARD = `[File: Morten - Nobly Kickoff - Opening and cue cards EN.docx](#media:${encodeURIComponent(SPACED)})`

describe('Office MEDIA paths with interior spaces', () => {
  const extensions = ['xlsb', 'xlsm', 'xltx', 'xltm', 'docm', 'dotx', 'pptm', 'potx', 'vsdx']
  const roots = ['C:/Users/example/scratch/', 'C:\\Users\\example\\scratch\\', '/tmp/', '~/Documents/']

  it('keeps the full Office file target without consuming adjacent prose or tags', () => {
    for (const extension of extensions) {
      for (const root of roots) {
        const name = `My Model v6.${extension}`
        const path = `${root}${name}`
        const card = `[File: ${name}](#media:${encodeURIComponent(path)})`

        expect(renderMediaTags(`MEDIA:${path}`)).toBe(card)
        expect(renderMediaTags(`Ready: MEDIA:${path} — enjoy`)).toBe(`Ready: ${card} — enjoy`)
        expect(mediaTagValues(`MEDIA:${path}MEDIA:/tmp/next.xlsx`)).toEqual([path, '/tmp/next.xlsx'])
        expect(renderMediaTags(`MEDIA:"${path}"`)).toBe(card)
      }
    }
  })

  it('settles streamed Office paths on the same complete file target', () => {
    for (const extension of extensions) {
      const path = `C:/Users/example/scratch/My Model v6.${extension}`
      let parts = appendAssistantTextPart([], 'Ready\nMEDIA:C:/Users/example/scratch/My')

      for (const chunk of [' Model v6.', extension, '\nDone']) {
        parts = appendAssistantTextPart(parts, chunk)
      }

      expect(chatMessageText({ id: 'office', parts, role: 'assistant' })).toBe(
        `Ready\n[File: My Model v6.${extension}](#media:${encodeURIComponent(path)})\nDone`
      )
    }
  })
})

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
