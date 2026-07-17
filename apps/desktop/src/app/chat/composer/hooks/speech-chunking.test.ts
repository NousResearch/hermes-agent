import { describe, expect, it } from 'vitest'

import { extractChunk } from './use-voice-conversation'

const take = (text: string, force = false) => extractChunk(text, force)

// The loop never hands the splitter a finished reply: it appends stream deltas
// to a buffer and drains whole sentences as they become available, then forces
// the remainder once the response completes. Testing a complete string hides
// every bug that only exists while the text is still arriving, so these feed the
// text in one character at a time and collect what would actually be spoken.
function speakStreamed(reply: string): string[] {
  let buffer = ''
  const spoken: string[] = []

  for (const character of reply) {
    buffer += character

    let chunk = take(buffer)

    while (chunk !== null) {
      spoken.push(chunk[0])
      buffer = chunk[1]
      chunk = take(buffer)
    }
  }

  const forced = take(buffer, true)

  if (forced && forced[0]) {
    spoken.push(forced[0])
  }

  return spoken
}

describe('extractChunk while the reply is still streaming', () => {
  // A '.' after a digit is ambiguous until the next character arrives: "1." is
  // either a finished sentence or the head of "1.2". Splitting on the buffer end
  // speaks "one" and then "two billion".
  it('keeps a decimal whole when the buffer ends mid-number', () => {
    expect(speakStreamed('Revenue grew to 1.2 billion. Margins held.')).toEqual([
      'Revenue grew to 1.2 billion.',
      'Margins held.'
    ])
    expect(speakStreamed('Version 2.0 shipped today. More below.')).toEqual([
      'Version 2.0 shipped today.',
      'More below.'
    ])
  })

  // The other half of the same trade: a sentence ending in a number still ends,
  // as soon as the space after it proves the number is finished.
  it('ends a sentence that finishes on a number', () => {
    expect(speakStreamed('It closed at 187.5. Volume was up today.')).toEqual([
      'It closed at 187.5.',
      'Volume was up today.'
    ])
    expect(speakStreamed('The answer is 42. Next question please.')).toEqual([
      'The answer is 42.',
      'Next question please.'
    ])
  })

  it('speaks a plain reply sentence by sentence', () => {
    expect(speakStreamed('This is a full sentence. And another one.')).toEqual([
      'This is a full sentence.',
      'And another one.'
    ])
  })

  it('keeps a list marker attached to its item', () => {
    expect(speakStreamed('1. First take the lid off. 2. Then pour.')).toEqual([
      '1. First take the lid off.',
      '2. Then pour.'
    ])
  })

  // The accepted cost of treating "42." and "2." as the same token: when every
  // item is shorter than the minimum, the next marker rides out on the previous
  // chunk. Audible only as an early boundary.
  it('glues a trailing marker when every list item is shorter than the minimum', () => {
    expect(speakStreamed('1. Yes. 2. No.')).toEqual(['1. Yes. 2.', 'No.'])
  })

  it('voices the tail once the response completes', () => {
    expect(speakStreamed('All done. No trailing stop')).toEqual(['All done.', 'No trailing stop'])
  })

  it('loses nothing across a whole reply', () => {
    const reply = 'First one here. Then 3.5 percent. 1. A point. Finally, the end.'

    expect(speakStreamed(reply).join(' ')).toBe(reply)
  })
})

describe('extractChunk', () => {
  it('waits rather than voicing a fragment with no boundary', () => {
    expect(take('a partial sentence with no end yet')).toBeNull()
  })

  it('voices what is left when forced', () => {
    expect(take('a partial sentence with no end yet', true)).toEqual([
      'a partial sentence with no end yet',
      ''
    ])
  })

  it('soft-wraps a long boundary-free run at a clause break', () => {
    const runOn = `${'x'.repeat(120)}, ${'y'.repeat(120)}`
    const result = take(runOn)

    expect(result).not.toBeNull()
    expect(result?.[0].length).toBeLessThan(runOn.length)
    expect(`${result?.[0]} ${result?.[1]}`).toBe(runOn)
  })

  // `。！？` end a sentence the same way `.` does. The boundary still needs
  // whitespace or the buffer end after it, so unspaced CJK runs stay whole, as
  // they did before this splitter, and MIN_SPEAK_CHARS counts characters, so
  // short CJK sentences merge.
  it('ends a sentence on CJK punctuation', () => {
    expect(take('これは長い日本語の文です。 次の文もあります。')).toEqual([
      'これは長い日本語の文です。',
      '次の文もあります。'
    ])
  })
})
