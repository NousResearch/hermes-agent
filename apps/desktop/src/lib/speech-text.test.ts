import { describe, expect, it } from 'vitest'

import { sanitizeTextForSpeech } from './speech-text'

describe('sanitizeTextForSpeech', () => {
  it('does not speak placeholders for fenced code blocks', () => {
    // The "code block omitted" summary used to be read aloud as English text
    // (#86602). Code that can't be spoken should be silence, not a sentence.
    // The "here is code:" colon also closes: the voice never waits on it.
    expect(sanitizeTextForSpeech('Here is code:\n```ts\nconst x = 1\n```\nDone.')).toBe(
      'Here is code. Done.'
    )
  })

  it('still keeps normal prose and inline code readable', () => {
    expect(sanitizeTextForSpeech('Use `git status` after the change.')).toBe('Use git status after the change.')
  })

  it('skips markdown table data while preserving surrounding human text', () => {
    const text = `Here is the quick takeaway: the totals remain unchanged.

| Item | Value | Notes |
| --- | ---: | --- |
| Example A | 10 | first row |
| Example B | 20 | second row |

Full detail stays visible on screen.`

    expect(sanitizeTextForSpeech(text)).toBe(
      'Here is the quick takeaway: the totals remain unchanged. Full detail stays visible on screen.'
    )
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

  it('does not speak a placeholder word for URLs', () => {
    // Used to say the English word "link" (#86602); URLs are silence now.
    expect(sanitizeTextForSpeech('See https://example.com/a-huge-page for details')).toBe(
      'See for details'
    )
  })

  it('expands symbols into words a voice can say', () => {
    const spoken = sanitizeTextForSpeech('~100 users, 2× RTX 5090, ≈50% sure, €5 each. Next → done.')

    expect(spoken).toContain('about 100 users')
    expect(spoken).toContain('2 times RTX 5090')
    expect(spoken).toContain('about 50 percent')
    expect(spoken).toContain('5 euros')
    expect(spoken).toContain('to done')
  })

  it('keeps ~~strike~~ readable instead of speaking tildes', () => {
    expect(sanitizeTextForSpeech('This ~~is~~ old.')).toBe('This is old.')
  })

  it('tames em dashes into speakable pauses', () => {
    // Real repro: "peek — the app password" produced an audible growl.
    // Lowercase after the dash is a comma pause; uppercase opens a sentence.
    // Digit-to-digit ranges become "to"; a dash next to a digit on one side
    // still gets tamed instead of surviving raw.
    expect(sanitizeTextForSpeech("The file's below if you want to peek — the app password is safe.")).toBe(
      "The file's below if you want to peek, the app password is safe."
    )
    expect(sanitizeTextForSpeech('You can peek — Done.')).toBe('You can peek. Done.')
    expect(sanitizeTextForSpeech('Mix red — green — blue.')).toBe('Mix red, green, blue.')
    expect(sanitizeTextForSpeech('Step 1 — open the file.')).toBe('Step 1, open the file.')
    expect(sanitizeTextForSpeech('See pages 5–10.')).toBe('See pages 5 to 10.')
  })

  it('expands trailing and bare euro signs', () => {
    // PT/ES convention writes the sign after the amount ("1.499,90 €").
    expect(sanitizeTextForSpeech('Costs 1.499,90 € per unit.')).toBe('Costs 1.499,90 euros per unit.')
    expect(sanitizeTextForSpeech('Prices in € are stable.')).toBe('Prices in euros are stable.')
  })

  it('closes a colon orphaned when its file link is stripped', () => {
    // Inline form: "below: MEDIA:/path" on one line. The link is stripped
    // mid-line, orphaning the colon at the end of the text. It must close.
    expect(sanitizeTextForSpeech('The file is below: MEDIA:/tmp/x.py')).toBe('The file is below.')
  })

  it('tames bare file paths into a spoken placeholder', () => {
    // No MEDIA: marker: "~/.config/himalaya/config.toml" in prose looped the
    // voice into "aaaa". Paths are screen addresses, not speech.
    expect(sanitizeTextForSpeech('check the config at ~/.config/himalaya/config.toml')).toBe(
      'check the config at the path'
    )
    expect(sanitizeTextForSpeech('read /etc/hosts for the mapping')).toBe('read the path for the mapping')
    expect(sanitizeTextForSpeech('open src/lib/app.ts to edit')).toBe('open the path to edit')
    // Not paths: slashed words, N/A, dates, decimal fractions, rates.
    expect(sanitizeTextForSpeech('pick A and/or B')).toBe('pick A and/or B')
    expect(sanitizeTextForSpeech('status is N/A')).toBe('status is N/A')
    expect(sanitizeTextForSpeech('due 2026/06/02')).toBe('due 2026/06/02')
    expect(sanitizeTextForSpeech('ratio 1.5/2.5')).toBe('ratio 1.5/2.5')
    expect(sanitizeTextForSpeech('pay 5/month')).toBe('pay 5/month')
    expect(sanitizeTextForSpeech('~100 users')).toBe('about 100 users')
  })

  it('closes a colon that a code block used to follow', () => {
    // The real repro: "one line added to the regex list:" then a code fence.
    // The voice hit the colon, found a wall of punctuation, and stuttered.
    expect(
      sanitizeTextForSpeech('One line added to the regex list:\n```ts\nconst x = 1\n```\nBye.')
    ).toBe('One line added to the regex list. Bye.')
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

  it('skips markdown tables without leading and trailing pipes', () => {
    const text = `Main takeaway: total is unchanged.

Item | Value
--- | ---:
Example A | 10
Example B | 20

Done.`

    expect(sanitizeTextForSpeech(text)).toBe('Main takeaway: total is unchanged. Done.')
  })

  it('skips markdown tables nested inside blockquotes', () => {
    const text = `Before the table.

> | Item | Value |
> | --- | ---: |
> | Example A | 10 |
> | Example B | 20 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. After the table.')
  })

  it('allows marker padding plus three spaces in blockquoted tables', () => {
    const text = `Before the table.

>    | Item | Value |
>    | --- | ---: |
>    | Example A | 10 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. After the table.')
  })

  it('skips explicit single-column markdown tables', () => {
    const text = `Before the table.

| Item |
| --- |
| Example A |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. After the table.')
  })

  it('preserves rows outside a table blockquote', () => {
    const text = `> | Item | Value |
> | --- | ---: |
> | Example A | 10 |
Outside | prose`

    expect(sanitizeTextForSpeech(text)).toBe('Outside | prose')
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

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. After the table.')
  })

  it('skips tables containing escaped pipe characters', () => {
    const text = `Before the table.

| Item \\| detail | Value |
| --- | ---: |
| Example A | 10 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. After the table.')
  })

  it('preserves indented code that resembles a table', () => {
    const text = `    Item | Value
    --- | ---
    Example A | 10`

    expect(sanitizeTextForSpeech(text)).toContain('Item | Value')
  })
})
