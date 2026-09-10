import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as I18nModule from '@/i18n'

import { sanitizeTextForSpeech } from './speech-text'

vi.mock('@/i18n', async importOriginal => {
  const actual = await importOriginal<typeof I18nModule>()

  return { ...actual, translateNow: vi.fn(actual.translateNow) }
})

const { setRuntimeI18nLocale, translateNow } = await import('@/i18n')
const actualI18n = await vi.importActual<typeof I18nModule>('@/i18n')

describe('sanitizeTextForSpeech', () => {
  afterEach(() => {
    setRuntimeI18nLocale('en')
    vi.mocked(translateNow).mockImplementation(actualI18n.translateNow)
  })

  it('summarizes fenced code blocks instead of reading them literally', () => {
    expect(sanitizeTextForSpeech('Here is code:\n```ts\nconst x = 1\n```\nDone.')).toBe(
      'Here is code: code block omitted Done.'
    )
  })

  it('speaks code blocks, links, and omitted tables in the active locale', () => {
    setRuntimeI18nLocale('zh')

    expect(sanitizeTextForSpeech('Here is code:\n```ts\nconst x = 1\n```\nDone.')).toBe('Here is code: 代码块已省略 Done.')
    expect(sanitizeTextForSpeech('See https://example.com for details.')).toBe('See 链接 for details.')

    const text = `Before the table.

| Item | Value |
| --- | ---: |
| Example A | 10 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. 表格已省略. After the table.')
  })

  it('treats a translated placeholder as literal text, not a $-pattern', () => {
    vi.mocked(translateNow).mockImplementation(key => (key === 'assistant.thread.speechCodeBlockOmitted' ? '$&' : ''))

    expect(sanitizeTextForSpeech('Here is code:\n```ts\nconst x = 1\n```\nDone.')).toBe('Here is code: $& Done.')
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
      'Here is the quick takeaway: the totals remain unchanged. table omitted. Full detail stays visible on screen.'
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

    expect(sanitizeTextForSpeech(text)).toBe('Main takeaway: total is unchanged. table omitted. Done.')
  })

  it('announces omitted markdown tables nested inside blockquotes', () => {
    const text = `Before the table.

> | Item | Value |
> | --- | ---: |
> | Example A | 10 |
> | Example B | 20 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. table omitted. After the table.')
  })

  it('allows marker padding plus three spaces in blockquoted tables', () => {
    const text = `Before the table.

>    | Item | Value |
>    | --- | ---: |
>    | Example A | 10 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. table omitted. After the table.')
  })

  it('announces omission of explicit single-column markdown tables', () => {
    const text = `Before the table.

| Item |
| --- |
| Example A |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. table omitted. After the table.')
  })

  it('preserves rows outside a table blockquote', () => {
    const text = `> | Item | Value |
> | --- | ---: |
> | Example A | 10 |
Outside | prose`

    expect(sanitizeTextForSpeech(text)).toBe('table omitted Outside | prose')
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

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. table omitted. After the table.')
  })

  it('skips tables containing escaped pipe characters', () => {
    const text = `Before the table.

| Item \\| detail | Value |
| --- | ---: |
| Example A | 10 |

After the table.`

    expect(sanitizeTextForSpeech(text)).toBe('Before the table. table omitted. After the table.')
  })

  it('preserves indented code that resembles a table', () => {
    const text = `    Item | Value
    --- | ---
    Example A | 10`

    expect(sanitizeTextForSpeech(text)).toContain('Item | Value')
  })
})
