import { describe, expect, it } from 'vitest'

import { cutSentences, IncrementalSpeechSentenceBuffer, sanitizeTextForSpeech } from './speech-text'

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

describe('cutSentences', () => {
  it('emits complete sentences and holds the incomplete tail', () => {
    const { sentences, rest } = cutSentences('This is the first full sentence. And then it keeps goi', false)

    expect(sentences).toEqual(['This is the first full sentence.'])
    expect(rest).toBe('And then it keeps goi')
  })

  it('buffers too-short fragments instead of firing per abbreviation', () => {
    const { sentences, rest } = cutSentences('e.g. it continues', false)

    expect(sentences).toEqual([])
    expect(rest).toBe('e.g. it continues')
  })

  it('flush drains everything including the tail', () => {
    const { sentences, rest } = cutSentences('First complete sentence right here. tail bit', true)

    expect(sentences).toEqual(['First complete sentence right here.', 'tail bit'])
    expect(rest).toBe('')
  })

  it('handles CJK terminators', () => {
    const { sentences } = cutSentences(
      '这是一个完整的中文句子，它的长度足够超过最小句子门槛，所以会被切分出来。 下一句',
      true
    )

    expect(sentences[0]).toContain('。')
    expect(sentences).toHaveLength(2)
  })

  it('cuts a short CJK opener alone when the backend sends tts.streaming.min_len', () => {
    const text = '记得，叫团团。 然后我们再说第二句话，这一句要长一些才行。 '

    // Historical 24-char floor (older backend, no key): the opener rides with sentence two.
    expect(cutSentences(text, false).sentences).toEqual(['记得，叫团团。 然后我们再说第二句话，这一句要长一些才行。'])
    // tts.streaming.min_len = 6 (the CJK voice setup from #96927): spoken on its own.
    expect(cutSentences(text, false, 6).sentences).toEqual([
      '记得，叫团团。',
      '然后我们再说第二句话，这一句要长一些才行。'
    ])
  })
})

describe('IncrementalSpeechSentenceBuffer', () => {
  it('does not hold later sentences when prose mentions a <thinking tag', () => {
    const buffer = new IncrementalSpeechSentenceBuffer()

    expect(buffer.append('Wrap the plan in a <thinking> tag first. ')).toEqual([
      'Wrap the plan in a <thinking> tag first.'
    ])
    expect(buffer.append('Then the answer follows here in prose. And a tail')).toEqual([
      'Then the answer follows here in prose.'
    ])
    expect(buffer.flush()).toEqual(['And a tail'])
  })
})
