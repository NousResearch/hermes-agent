import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'
import {
  createIncrementalMarkdownPreprocessor,
  selectMarkdownPreprocessor
} from './markdown-preprocess-cache'
import { createIncrementalPreprocessWithTailRepair } from './markdown-preprocess-wrapper'

const LONG_SETTLED_PROSE = Array.from(
  { length: 48 },
  (_, index) => `Paragraph ${index}: streaming markdown keeps completed prose stable across later chunks.\n\n`
).join('')

const STATEFUL_TAILS = [
  ['backtick fence', '```ts\nconst value = `streaming`;\n```\nAfter code.'],
  ['tilde fence', '~~~math\nx^2 + y^2\n~~~\nAfter math.'],
  ['inline math', 'Inline $x + \\sqrt[3]{8}$, unfinished $y + z, and currency $19.99.'],
  ['display math', 'Display:\n$$\n\\begin{aligned}\na&=b\\\\\nc&=d\n\\end{aligned}\n$$\nThen \\[e=f\\].'],
  ['links', '[remote](https://example.com/a) https://example.com/live [file](/tmp/report.md)'],
  [
    'directives',
    '::chart{series="one"}\n\n@session:work/20260831_deadbeef\n\n[Preview: report](#preview:/tmp/report.md)'
  ],
  ['HTML and reasoning', '<think>private stream</think><section><div>visible</div></section><article>open'],
  ['citation', 'A sourced claim[12] followed by [label](#target) and an incomplete [link'],
  ['whitespace normalization', 'Trailing spaces   \n\n\nFinal paragraph.']
] as const

describe('createIncrementalMarkdownPreprocessor', () => {
  it('uses one normal pass for completed text instead of priming incremental state', () => {
    let completedPasses = 0
    let incrementalPasses = 0

    const completed = (text: string) => {
      completedPasses += 1

      return preprocessMarkdown(text)
    }

    const incremental = (text: string) => {
      incrementalPasses += 1

      return preprocessMarkdown(text)
    }

    const source = `${LONG_SETTLED_PROSE}A completed answer.`

    expect(selectMarkdownPreprocessor(false, incremental, completed)(source)).toBe(preprocessMarkdown(source))
    expect(completedPasses).toBe(1)
    expect(incrementalPasses).toBe(0)
  })

  it('does not preprocess a cacheable first value twice', () => {
    const processedLengths: number[] = []

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedLengths.push(text.length)

      return preprocessMarkdown(text)
    })

    const source = `${LONG_SETTLED_PROSE}Newest paragraph is still changing.`

    expect(incremental(source)).toBe(preprocessMarkdown(source))
    expect(processedLengths.reduce((total, length) => total + length, 0)).toBeLessThanOrEqual(source.length)
  })

  it('releases retained streaming prefixes when cleared', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const source = `${LONG_SETTLED_PROSE}Newest paragraph is still changing.`

    incremental(source)
    incremental(source)
    incremental.clear()
    const beforeFreshPass = processedCharacters

    expect(incremental(source)).toBe(preprocessMarkdown(source))
    expect(processedCharacters - beforeFreshPass).toBeGreaterThanOrEqual(source.length)
  })

  it('keeps independent append lineages hot when message surfaces render interleaved', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const firstPrefix = LONG_SETTLED_PROSE.replaceAll('Paragraph', 'First paragraph')
    const secondPrefix = LONG_SETTLED_PROSE.replaceAll('Paragraph', 'Second paragraph')
    const first = `${firstPrefix}First live tail`
    const second = `${secondPrefix}Second live tail`

    incremental(first)
    incremental(second)
    const workBeforeFirstAppend = processedCharacters
    const appendedFirst = `${first} plus another token`

    expect(incremental(appendedFirst)).toBe(preprocessMarkdown(appendedFirst))
    expect(processedCharacters - workBeforeFirstAppend).toBeLessThan(appendedFirst.length / 4)
  })

  it('keeps five interleaved append lineages hot', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const lineages = Array.from({ length: 5 }, (_, index) => {
      const prefix = LONG_SETTLED_PROSE.replaceAll('Paragraph', `Lineage ${index} paragraph`)

      return `${prefix}Live tail ${index}`
    })

    for (const text of lineages) {
      expect(incremental(text)).toBe(preprocessMarkdown(text))
    }

    let appendedCharacters = 0

    for (let round = 0; round < 2; round += 1) {
      const workBeforeRound = processedCharacters

      for (let index = 0; index < lineages.length; index += 1) {
        const text = lineages[index]!
        const appended = `${text} plus another token ${round}`
        lineages[index] = appended
        appendedCharacters += appended.length - text.length

        expect(incremental(appended)).toBe(preprocessMarkdown(appended))
      }

      if (round === 1) {
        expect(processedCharacters - workBeforeRound).toBeLessThan(appendedCharacters * 5)
      }
    }
  })

  it('accounts for wrapper scans across representative streaming updates', () => {
    let wrapperScannedCharacters = 0
    let preprocessedCharacters = 0

    const incremental = createIncrementalPreprocessWithTailRepair(
      text => {
        preprocessedCharacters += text.length

        return preprocessMarkdown(text)
      },
      text => {
        wrapperScannedCharacters += text.length

        return text
      }
    )

    const paragraphs = Array.from(
      { length: 120 },
      (_, index) => `Transcript paragraph ${index}: a completed answer remains stable while the final sentence grows.\n\n`
    )

    let text = paragraphs.join('')
    let fullPreprocessCharacters = text.length

    expect(incremental(text)).toBe(preprocessMarkdown(text))

    for (let update = 0; update < 40; update += 1) {
      const suffix = `Update ${update} adds another sentence to the live answer. `
      text += suffix
      fullPreprocessCharacters += text.length

      expect(incremental(text)).toBe(preprocessMarkdown(text))
    }

    expect(preprocessedCharacters).toBeLessThan(fullPreprocessCharacters / 8)
    expect(wrapperScannedCharacters).toBeLessThanOrEqual(fullPreprocessCharacters)
  })

  it('reuses a settled prefix discovered inside the first observed value', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const first = `${LONG_SETTLED_PROSE}The live tail has already started`
    const second = `${first}, and another token arrived.`

    expect(incremental(first)).toBe(preprocessMarkdown(first))
    const workAfterFirstValue = processedCharacters

    expect(incremental(second)).toBe(preprocessMarkdown(second))
    expect(processedCharacters - workAfterFirstValue).toBeLessThan(second.length / 4)
  })

  it('discovers settled prose before an incomplete construct in the first observed value', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const first = `${LONG_SETTLED_PROSE}\`\`\`ts\nconst streamed = "open"`
    const second = `${first}\nconst next = true`

    expect(incremental(first)).toBe(preprocessMarkdown(first))
    const workAfterFirstValue = processedCharacters

    expect(incremental(second)).toBe(preprocessMarkdown(second))
    expect(processedCharacters - workAfterFirstValue).toBeLessThan(second.length / 4)
  })

  it('keeps blank-line normalization byte-identical as extra newlines arrive', () => {
    const incremental = createIncrementalMarkdownPreprocessor()

    expect(incremental(LONG_SETTLED_PROSE)).toBe(preprocessMarkdown(LONG_SETTLED_PROSE))

    const withTrailingBlankLine = `${LONG_SETTLED_PROSE}\n`
    expect(incremental(withTrailingBlankLine)).toBe(preprocessMarkdown(withTrailingBlankLine))

    const withFollowingParagraph = `${withTrailingBlankLine}Following prose.`
    expect(incremental(withFollowingParagraph)).toBe(preprocessMarkdown(withFollowingParagraph))
  })

  it('stays byte-identical while adversarial constructs stream after a settled prefix', () => {
    const incremental = createIncrementalMarkdownPreprocessor()

    const chunks = [
      'A live paragraph starts',
      ' and then gains a citation',
      '[12].\n\n',
      'Visit https://exa',
      'mple.com/path?q=1.\n\n',
      'Use `incomplete',
      ' code` here.\n\n',
      '```ts\nconst fence = "open"',
      '\n```\n\n',
      '~~~math\nx^2 + y^2',
      '\n~~~\n\n',
      'Inline math $x +',
      ' y$ and display math:\n$$\na+b',
      '\n$$\n\n',
      '<think>private reasoning',
      '</think>Visible answer.\n\n',
      '<section><div>raw HTML',
      '</div></section>\n\n',
      '[Preview: report](',
      '#preview:/tmp/report.md)\n\n',
      'See @session:work/',
      '20260831_abc123 for context.\n\n',
      '[report](/tmp/report.md)\n\n',
      '\\[a+b',
      '\\]\n\n',
      '\n',
      '\nFinal paragraph.'
    ]

    let text = LONG_SETTLED_PROSE

    for (const chunk of chunks) {
      text += chunk

      expect(incremental(text), `after appending ${JSON.stringify(chunk)}`).toBe(preprocessMarkdown(text))
    }
  })

  it.each(STATEFUL_TAILS)('matches full preprocessing at every %s append boundary', (_label, tail) => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    let text = LONG_SETTLED_PROSE

    expect(incremental(text)).toBe(preprocessMarkdown(text))
    const workAfterPrime = processedCharacters
    let fullPreprocessCharacters = 0

    for (const character of tail) {
      text += character
      fullPreprocessCharacters += text.length

      expect(incremental(text), `after appending ${JSON.stringify(character)}`).toBe(preprocessMarkdown(text))
    }

    expect(processedCharacters - workAfterPrime).toBeLessThan(fullPreprocessCharacters / 4)
  })

  it.each([
    ['backtick fence', '```ts\nconst value = 1'],
    ['tilde fence', '~~~ts\nconst value = 1'],
    ['math', '$x + y$'],
    ['raw URL', 'Visit https://example.com/path?q=1'],
    ['raw HTML and reasoning', '<think>hidden</think><div>visible'],
    ['citation and preview marker', 'Claim[1]. [Preview: x](#preview:/tmp/x)'],
    ['backslash math delimiter', '\\[x + y\\]']
  ])('keeps reusing a settled prefix when a later append introduces %s', (_label, suffix) => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    incremental(LONG_SETTLED_PROSE)
    const workBeforeUnsafeAppend = processedCharacters
    const text = LONG_SETTLED_PROSE + suffix

    expect(incremental(text)).toBe(preprocessMarkdown(text))
    expect(processedCharacters - workBeforeUnsafeAppend).toBeLessThan(text.length / 4)
  })

  it('incrementalizes plain prose containing complete session refs', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const transformedPrefix = Array.from(
      { length: 56 },
      (_, index) => `See @session:default/20260831_session${index} for details.\n\n`
    ).join('')

    expect(incremental(transformedPrefix)).toBe(preprocessMarkdown(transformedPrefix))
    const workBeforeAppend = processedCharacters
    const appended = `${transformedPrefix}A final plain paragraph is still streaming.`

    expect(incremental(appended)).toBe(preprocessMarkdown(appended))
    expect(processedCharacters - workBeforeAppend).toBeLessThan(appended.length / 4)
  })

  it('preserves link and HTML-depth rewrites in an incrementally reprocessed tail', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const settledPrefix = LONG_SETTLED_PROSE.repeat(10)
    const nestedHtml = `${'<div>'.repeat(280)}https://example.com/deep${'</div>'.repeat(280)}`
    const initial = `${settledPrefix}${nestedHtml}`

    expect(incremental(initial)).toBe(preprocessMarkdown(initial))
    const workBeforeAppend = processedCharacters
    const appended = `${initial}\nAnother link: https://example.com/final`

    expect(incremental(appended)).toBe(preprocessMarkdown(appended))
    expect(processedCharacters - workBeforeAppend).toBeLessThan(appended.length / 4)
  })

  it('does not settle an inline tilde run that a later append can close around transformed prose', () => {
    const incremental = createIncrementalMarkdownPreprocessor()
    const first = `${LONG_SETTLED_PROSE}Inline ~~~ opener.\n\nBridge one.\n\nBridge two.\n\nhttps://example.com/live`

    expect(incremental(first)).toBe(preprocessMarkdown(first))

    const closed = `${first} and later ~~~`
    expect(incremental(closed)).toBe(preprocessMarkdown(closed))
  })

  it('matches full preprocessing while randomized plain-prose streams keep reusing the prefix', () => {
    const fragments = [
      'plain words ',
      'https://exa',
      'mple.com/path ',
      '@session:default/',
      '20260831_deadbeef ',
      'citation-free prose. ',
      '\n',
      '\n\n',
      '   ',
      'punctuation: one, two; three! '
    ]

    for (let seed = 1; seed <= 16; seed += 1) {
      const incremental = createIncrementalMarkdownPreprocessor()
      let state = seed
      let text = LONG_SETTLED_PROSE

      expect(incremental(text)).toBe(preprocessMarkdown(text))

      for (let step = 0; step < 100; step += 1) {
        state = (state * 1103515245 + 12345) >>> 0
        text += fragments[state % fragments.length]

        expect(incremental(text), `plain seed ${seed}, step ${step}`).toBe(preprocessMarkdown(text))
      }
    }
  })

  it('matches full preprocessing for deterministic randomized append streams', () => {
    const fragments = [
      'plain words ',
      '\n\n',
      '\n',
      '`',
      '```js\n',
      '```\n',
      '~~~\n',
      '$',
      '$$\n',
      '\\[',
      '\\]',
      '<think>',
      '</think>',
      '<div>',
      '</div>',
      'https://example.com/a',
      '[7]',
      '[Preview: x](#preview:/tmp/x)',
      '@session:default/20260831_deadbeef',
      '[file](/tmp/a.md)'
    ]

    for (let seed = 1; seed <= 24; seed += 1) {
      const incremental = createIncrementalMarkdownPreprocessor()
      let state = seed
      let text = LONG_SETTLED_PROSE

      for (let step = 0; step < 80; step += 1) {
        state = (state * 1664525 + 1013904223) >>> 0
        text += fragments[state % fragments.length]

        expect(incremental(text), `seed ${seed}, step ${step}`).toBe(preprocessMarkdown(text))
      }
    }
  })

  it.each([
    ['backtick fence', '```ts\n'],
    ['tilde fence', '~~~ts\n'],
    ['math', '$x$\n'],
    ['raw HTML', '<div>\n'],
    ['markdown or citation', '[1]\n'],
    ['backslash delimiter', '\\[x\\]\n']
  ])('falls back to full preprocessing when the prefix contains %s state', (_label, opener) => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const first = `${opener}${'unsafe prefix text '.repeat(180)}\n\nTail one`
    const second = `${first} plus two`

    expect(incremental(first)).toBe(preprocessMarkdown(first))
    const workAfterFirstValue = processedCharacters
    expect(incremental(second)).toBe(preprocessMarkdown(second))
    expect(processedCharacters - workAfterFirstValue).toBe(second.length)
  })

  it('falls back on non-append edits instead of reusing stale output', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const original = `${LONG_SETTLED_PROSE}Original tail`
    const edited = `${LONG_SETTLED_PROSE.replace('Paragraph 0', 'Edited paragraph')}Edited tail`

    incremental(original)
    const workBeforeEdit = processedCharacters

    expect(incremental(edited)).toBe(preprocessMarkdown(edited))
    expect(processedCharacters - workBeforeEdit).toBeGreaterThanOrEqual(edited.length)
  })

  it.each([
    ['tail edit', `${LONG_SETTLED_PROSE}Original streaming tail`, `${LONG_SETTLED_PROSE}Replaced streaming tail`],
    ['truncation', `${LONG_SETTLED_PROSE}Original streaming tail`, LONG_SETTLED_PROSE]
  ])('resets settled-prefix reuse after a non-append %s', (_label, original, changed) => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    incremental(original)
    const workBeforeChange = processedCharacters

    expect(incremental(changed)).toBe(preprocessMarkdown(changed))
    expect(processedCharacters - workBeforeChange).toBeGreaterThanOrEqual(changed.length)
  })

  it('does not retain append state beyond the renderer markdown size bound', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const paragraph = 'Bounded retained markdown state stays finite.\n\n'
    const oversized = paragraph.repeat(Math.ceil(200_001 / paragraph.length))

    expect(incremental(oversized)).toBe(preprocessMarkdown(oversized))
    const workBeforeAppend = processedCharacters
    const appended = `${oversized}x`

    expect(incremental(appended)).toBe(preprocessMarkdown(appended))
    expect(processedCharacters - workBeforeAppend).toBeGreaterThanOrEqual(appended.length)
  })

  it('keeps append-stream preprocessing work sublinear in accumulated input', () => {
    let processedCharacters = 0

    const incremental = createIncrementalMarkdownPreprocessor(text => {
      processedCharacters += text.length

      return preprocessMarkdown(text)
    })

    const paragraphs = Array.from(
      { length: 160 },
      (_, index) => `Paragraph ${index}: streaming markdown stays byte-identical while settled prose is reused.\n\n`
    )

    let text = ''
    let fullPreprocessCharacters = 0

    for (const paragraph of paragraphs) {
      text += paragraph
      fullPreprocessCharacters += text.length

      expect(incremental(text)).toBe(preprocessMarkdown(text))
    }

    // A deterministic work-count benchmark: count characters handed to the
    // full preprocessor rather than relying on noisy wall-clock timings.
    expect(processedCharacters).toBeLessThan(fullPreprocessCharacters / 8)
  })
})
