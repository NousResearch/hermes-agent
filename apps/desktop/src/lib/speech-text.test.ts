import { describe, expect, it } from 'vitest'

import { IncrementalSpeechSentenceBuffer, sanitizeTextForSpeech } from './speech-text'

describe('sanitizeTextForSpeech', () => {
  it('does not speak placeholders for fenced code blocks', () => {
    // The "code block omitted" summary used to be read aloud as English text
    // (#86602). Code that can't be spoken should be silence, not a sentence.
    // The "here is code:" colon also closes: the voice never waits on it.
    expect(sanitizeTextForSpeech('Here is code:\n```ts\nconst x = 1\n```\nDone.')).toBe('Here is code. Done.')
  })

  it('still keeps normal prose and inline code readable', () => {
    expect(sanitizeTextForSpeech('Use `git status` after the change.')).toBe('Use git status after the change.')
  })

  it('reads the table header, not its data, between surrounding human text', () => {
    const text = `Here is the quick takeaway: the totals remain unchanged.

| Item | Value | Notes |
| --- | ---: | --- |
| Example A | 10 | first row |
| Example B | 20 | second row |

Full detail stays visible on screen.`

    expect(sanitizeTextForSpeech(text)).toBe(
      'Here is the quick takeaway: the totals remain unchanged. Item, Value, Notes. Full detail stays visible on screen.'
    )
  })

  it('speaks a non-English reply with no English words injected (#86602)', () => {
    // Placeholders used to follow the code, not the reply's language: a
    // Chinese voice read "code block omitted" / "link" and skipped tables
    // entirely. Code and URLs are silence; the table header is read in the
    // reply's own language.
    const text = `对比如下：

| 模型 | 价格 |
| --- | ---: |
| 甲 | 10 |

代码：
\`\`\`py
print(1)
\`\`\`
详情见 https://example.com/docs`

    const spoken = sanitizeTextForSpeech(text)

    expect(spoken).toContain('模型, 价格')
    expect(spoken).not.toMatch(/[A-Za-z]/)
  })

  it('stays silent for a table whose header cells are all empty', () => {
    const text = `Before the table.

|   |   |
| --- | --- |
| a | b |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. After the table.')
  })

  it('does not strip prose that merely contains a pipe character', () => {
    const text = 'Use the summary first | keep the table on screen when it matters.'

    expect(sanitizeTextForSpeech(text)).toBe('Use the summary first | keep the table on screen when it matters.')
  })

  it('does not duplicate punctuation across paragraph breaks', () => {
    const text = `First sentence.

Second sentence.`

    expect(sanitizeTextForSpeech(text)).toBe('First sentence. Second sentence.')
  })

  it('does not speak MEDIA file-link tokens', () => {
    // Rendering shows these as "Open inference-server-shopping-list.xlsx";
    // the hyphenated slug + odd extension made the voice loop ("eeeeee").
    const text = 'The files are below.\nMEDIA:/Users/ricardo/Documents/inference-server-shopping-list.xlsx\nBye.'

    expect(sanitizeTextForSpeech(text)).toBe('The files are below. Bye.')
  })

  it('keeps the sentence break after an inline MEDIA token', () => {
    expect(sanitizeTextForSpeech('See MEDIA:/tmp/report-2026-q3.xlsx. Then reply.')).toBe('See. Then reply.')
  })

  it('does not speak a placeholder word for URLs', () => {
    // Used to say the English word "link" (#86602); URLs are silence now.
    expect(sanitizeTextForSpeech('See https://example.com/a-huge-page for details')).toBe('See for details')
  })

  it('keeps ~~strike~~ readable instead of speaking tildes', () => {
    expect(sanitizeTextForSpeech('This ~~is~~ old.')).toBe('This is old.')
  })

  it('closes a colon orphaned when its file link is stripped', () => {
    // Inline form: "below: MEDIA:/path" on one line. The link is stripped
    // mid-line, orphaning the colon at the end of the text. It must close.
    expect(sanitizeTextForSpeech('The file is below: MEDIA:/tmp/x.py')).toBe('The file is below.')
  })

  it('closes a colon that a code block used to follow', () => {
    // The real repro: "one line added to the regex list:" then a code fence.
    // The voice hit the colon, found a wall of punctuation, and stuttered.
    expect(sanitizeTextForSpeech('One line added to the regex list:\n```ts\nconst x = 1\n```\nBye.')).toBe(
      'One line added to the regex list. Bye.'
    )
  })

  it('closes a colon that ends the speakable text', () => {
    expect(sanitizeTextForSpeech('The regex list:')).toBe('The regex list.')
  })

  it.each([
    ['markdown emphasis', '**First sentence.**\n\nSecond sentence.', 'First sentence. Second sentence.'],
    ['a closing quote', '“First sentence.”\n\nSecond sentence.', '“First sentence.” Second sentence.'],
    ['a closing parenthesis', '(First sentence.)\n\nSecond sentence.', '(First sentence.) Second sentence.']
  ])('does not duplicate punctuation after %s', (_label, text, expected) => {
    expect(sanitizeTextForSpeech(text)).toBe(expected)
  })

  it('reads only the header of markdown tables without leading and trailing pipes', () => {
    const text = `Main takeaway: total is unchanged.

Item | Value
--- | ---:
Example A | 10
Example B | 20

Done.`

    expect(sanitizeTextForSpeech(text)).toBe('Main takeaway: total is unchanged. Item, Value. Done.')
  })

  it('reads only the header of markdown tables nested inside blockquotes', () => {
    const text = `Before the table.

> | Item | Value |
> | --- | ---: |
> | Example A | 10 |
> | Example B | 20 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. Item, Value. After the table.')
  })

  it('allows marker padding plus three spaces in blockquoted tables', () => {
    const text = `Before the table.

>    | Item | Value |
>    | --- | ---: |
>    | Example A | 10 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. Item, Value. After the table.')
  })

  it('reads only the header of explicit single-column markdown tables', () => {
    const text = `Before the table.

| Item |
| --- |
| Example A |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. Item. After the table.')
  })

  it('preserves rows outside a table blockquote', () => {
    const text = `> | Item | Value |
> | --- | ---: |
> | Example A | 10 |
Outside | prose`

    expect(sanitizeTextForSpeech(text)).toBe('Item, Value. Outside | prose')
  })

  it('preserves malformed tables with mismatched column counts', () => {
    const text = `Heading | Detail
--- | --- | ---
Keep this prose.`

    expect(sanitizeTextForSpeech(text)).toContain('Heading | Detail')
  })

  it('skips GFM body rows whose cell counts differ from the header', () => {
    const text = `Before the table.

| Item | Value |
| --- | ---: |
| Example A |
| Example B | 20 | ignored |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. Item, Value. After the table.')
  })

  it('reads headers containing escaped pipe characters', () => {
    const text = `Before the table.

| Item \\| detail | Value |
| --- | ---: |
| Example A | 10 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. Item detail, Value. After the table.')
  })

  it('preserves indented code that resembles a table', () => {
    const text = `    Item | Value
    --- | ---
    Example A | 10`

    expect(sanitizeTextForSpeech(text)).toContain('Item | Value')
  })
})

describe('IncrementalSpeechSentenceBuffer', () => {
  it('emits completed sentences while retaining the unfinished tail', () => {
    const buffer = new IncrementalSpeechSentenceBuffer()

    expect(buffer.append('The first sentence is ready. The second')).toEqual(['The first sentence is ready.'])
    expect(buffer.append(' sentence is ready too. ')).toEqual(['The second sentence is ready too.'])
    expect(buffer.flush()).toEqual([])
  })

  it('keeps short fragments with the following sentence', () => {
    const buffer = new IncrementalSpeechSentenceBuffer()

    expect(buffer.append('Hi! This longer sentence is ready. ')).toEqual(['Hi! This longer sentence is ready.'])
  })

  it('does not emit a reasoning block split across deltas', () => {
    const buffer = new IncrementalSpeechSentenceBuffer()

    expect(buffer.append('<think>Private reason.')).toEqual([])
    expect(buffer.append('</think> The public answer is ready. ')).toEqual(['The public answer is ready.'])
  })
})
