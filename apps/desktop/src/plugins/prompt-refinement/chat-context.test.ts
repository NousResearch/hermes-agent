import { describe, expect, it } from 'vitest'

import {
  CONTEXT_CHARS_EACH,
  CONTEXT_CHARS_TOTAL,
  CONTEXT_MESSAGES,
  formatContext,
  recentContext,
  registerContextReader,
  type TranscriptLine
} from './chat-context'

describe('chat context excerpt', () => {
  it('labels speakers, oldest first, with whitespace collapsed', () => {
    const lines: TranscriptLine[] = [
      { role: 'user', text: 'Fix the parser in\n\n src/parser.ts' },
      { role: 'assistant', text: 'Done. I split it into   three modules.' }
    ]

    expect(formatContext(lines)).toBe(
      'User: Fix the parser in src/parser.ts\nAssistant: Done. I split it into three modules.'
    )
  })

  it('clips each message and keeps the newest when the total runs over', () => {
    const long = 'x'.repeat(CONTEXT_CHARS_EACH * 2)
    const lines: TranscriptLine[] = Array.from({ length: 12 }, (_, i) => ({ role: 'user', text: `${i} ${long}` }))
    const out = formatContext(lines)

    expect(out.length).toBeLessThanOrEqual(CONTEXT_CHARS_TOTAL)
    out.split('\n').forEach(line => expect(line.length).toBeLessThanOrEqual('User: '.length + CONTEXT_CHARS_EACH))
    expect(out.split('\n').at(-1)).toMatch(/^User: 11 /)
  })

  it('reads only the composer it was asked for, a bounded number of messages', () => {
    const asked: number[] = []

    const release = registerContextReader('tile:b', limit => {
      asked.push(limit)

      return [{ role: 'user', text: 'from b' }]
    })

    expect(recentContext('main')).toBe('')
    expect(recentContext('tile:b')).toBe('User: from b')
    expect(asked).toEqual([CONTEXT_MESSAGES])

    release()
    expect(recentContext('tile:b')).toBe('')
  })

  it('a reader that throws leaves the refinement without context', () => {
    const release = registerContextReader('main', () => {
      throw new Error('transcript gone')
    })

    expect(recentContext('main')).toBe('')
    release()
  })

  it('a stale release does not remove a newer reader for the same composer', () => {
    const releaseOld = registerContextReader('main', () => [{ role: 'user', text: 'old' }])
    const releaseNew = registerContextReader('main', () => [{ role: 'user', text: 'new' }])

    releaseOld()
    expect(recentContext('main')).toBe('User: new')
    releaseNew()
  })
})
