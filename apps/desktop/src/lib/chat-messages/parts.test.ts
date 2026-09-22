// Regression for #96657: an unquoted MEDIA: path with interior spaces rendered
// a card for the text before the first space and left the rest as prose.
import { describe, expect, it } from 'vitest'

import { appendAssistantTextPart, chatMessageText, mediaTagValues, renderMediaTags } from './parts'

const SPACED = '/home/hermes/Morten - Nobly Kickoff - Opening and cue cards EN.docx'
const CARD = `[File: Morten - Nobly Kickoff - Opening and cue cards EN.docx](#media:${encodeURIComponent(SPACED)})`
const SPACED_LATEX = '/home/hermes/research notes/draft (final).tex'
const LATEX_CARD = `[File: draft (final).tex](#media:${encodeURIComponent(SPACED_LATEX)})`

describe('renderMediaTags with interior spaces', () => {
  it('keeps the whole spaced path in one card on every surface that reads MEDIA tags', () => {
    expect(renderMediaTags(`MEDIA:${SPACED}`)).toBe(CARD)
    expect(renderMediaTags(`MEDIA:${SPACED_LATEX}`)).toBe(LATEX_CARD)
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
