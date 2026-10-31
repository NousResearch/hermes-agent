import { describe, expect, it } from 'vitest'

import { isLikelyProseCodeBlock } from '@/lib/markdown-code'
import { normalizeFilePreviewMath, preprocessMarkdown } from '@/lib/markdown-preprocess'

// Listings whose own text carries fence characters: a ``` mid-line, and the
// inner fences of a markdown example fenced with a longer or other marker.
// Each must reach the renderer byte-for-byte, with the prose after it outside.
const LISTINGS = [
  ['```python', 'def is_fence(line):', '    return line.startswith("```")', 'first = parts[0]', 'cost = "$5"', '```'],
  ['````markdown', '```js', 'const a = arr[0];', '```', '````'],
  ['~~~md', '```sh', 'export A=$5; echo \\(x\\)', '```', '~~~'],
  ['```js', 'const tilde = "~~~ and $5 [1]"', '```']
].map(lines => lines.join('\n'))

describe('fenced code segmentation', () => {
  it('keeps prose-and-PowerShell instructions as a literal listing (#145379)', () => {
    const body = [
      'Runda 3, czesc 1. Zmieniasz JEDNA rzecz: usun pin interpretera.',
      'NIE ruszaj .venv, NIE uruchamiaj hermes update.',
      `"STARY PIN = " + [Environment]::GetEnvironmentVariable('HERMES_DESKTOP_PYTHON','User')`,
      `[Environment]::SetEnvironmentVariable('HERMES_DESKTOP_PYTHON',$null,'User')`,
      'Zamelduj wynik i CZEKAJ.'
    ].join('\n')

    const listing = ['```', body, '```'].join('\n')

    expect(preprocessMarkdown(listing)).toBe(listing)
    expect(isLikelyProseCodeBlock('', body)).toBe(false)
  })

  it('keeps the same listing fenced under every plain-text language tag', () => {
    const body = [
      'Run exactly this command.',
      'Keep the interpreter unchanged.',
      "[Environment]::SetEnvironmentVariable('HERMES_DESKTOP_PYTHON',$null,'User')",
      'Report the result and wait.'
    ].join('\n')

    for (const language of ['text', 'plain', 'plaintext', 'md', 'markdown']) {
      const listing = ['```' + language, body, '```'].join('\n')

      expect(preprocessMarkdown(listing)).toBe(listing)
      expect(isLikelyProseCodeBlock(language, body)).toBe(false)
    }
  })

  it('keeps a bullet list that mixes prose and shell variables fenced', () => {
    const body = ['- Run exactly this command.', 'echo $HOME', '- Report the result and wait.'].join('\n')
    const listing = ['```', body, '```'].join('\n')

    expect(preprocessMarkdown(listing)).toBe(listing)
    expect(isLikelyProseCodeBlock('', body)).toBe(false)
  })

  it.each(['echo $HOME', 'a `code` span', 'X::Y path segment', '"pin = " + value'])(
    'keeps a listing containing %s fenced',
    signal => {
      const body = [
        'Run exactly this command.',
        'Keep the interpreter unchanged.',
        signal,
        'Report the result and wait.'
      ].join('\n')

      const listing = ['```', body, '```'].join('\n')

      expect(preprocessMarkdown(listing)).toBe(listing)
      expect(isLikelyProseCodeBlock('', body)).toBe(false)
    }
  )

  it('still unwraps a plain-language paragraph fence', () => {
    const body = [
      'This is the first sentence of a paragraph.',
      'Here is a second line that continues the thought.',
      'And a third concluding line follows here.'
    ].join('\n')

    const output = preprocessMarkdown(['```', body, '```'].join('\n'))

    expect(output).not.toContain('```')
    expect(output).toContain('This is the first sentence of a paragraph.')
    expect(isLikelyProseCodeBlock('', body)).toBe(true)
  })

  it('keeps a listing that contains fence characters intact in chat markdown', () => {
    for (const listing of LISTINGS) {
      const output = preprocessMarkdown(`Example:\n\n${listing}\n\nAfter the example.`)

      expect(output).toContain(listing)
      expect(output.endsWith(`${listing}\n\nAfter the example.`)).toBe(true)
    }
  })

  it('keeps the same listings intact in a file preview, CRLF included', () => {
    for (const listing of [...LISTINGS, '```sh\r\nexport A=$5\r\n```']) {
      const input = `Intro \\(y\\)\n\n${listing}\n\nOutro`

      expect(normalizeFilePreviewMath(input)).toBe(`Intro $y$\n\n${listing}\n\nOutro`)
    }
  })
})
