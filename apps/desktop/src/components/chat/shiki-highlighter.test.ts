import { describe, expect, it } from 'vitest'

import { chunkByLines, copyableCodeText, exceedsHighlightBudget } from '@/components/chat/shiki-highlighter'

describe('exceedsHighlightBudget', () => {
  it('highlights normal-sized blocks', () => {
    expect(exceedsHighlightBudget('const x = 1\n'.repeat(100))).toBe(false)
  })

  it('skips highlighting past the line budget', () => {
    expect(exceedsHighlightBudget('x\n'.repeat(5_000))).toBe(true)
  })

  it('skips highlighting past the char budget on few lines', () => {
    expect(exceedsHighlightBudget('a'.repeat(200_000))).toBe(true)
  })

  it('short-circuits on char budget before line loop', () => {
    expect(exceedsHighlightBudget('y\n'.repeat(250_000))).toBe(true)
  })
})

describe('chunkByLines', () => {
  it('keeps a small block as a single chunk', () => {
    const code = 'a\nb\nc'
    expect(chunkByLines(code, 200)).toEqual([{ text: code, lines: 3 }])
  })

  it('splits a large block and reconstructs it losslessly', () => {
    const code = Array.from({ length: 1000 }, (_, i) => `line ${i}`).join('\n')
    const chunks = chunkByLines(code, 200)

    expect(chunks).toHaveLength(5)
    expect(chunks.map(chunk => chunk.text).join('\n')).toBe(code)
    expect(chunks.reduce((sum, chunk) => sum + chunk.lines, 0)).toBe(1000)
  })
})


describe('copyableCodeText', () => {
  it('unwraps quoted URL reference directives before copying a code block', () => {
    expect(copyableCodeText('curl @url:`https://example.com/image.png` -o image.png')).toBe(
      'curl https://example.com/image.png -o image.png'
    )
  })

  it('unwraps bare URL reference directives without losing code punctuation', () => {
    expect(copyableCodeText('curl @url:https://example.com/image.png; echo done')).toBe(
      'curl https://example.com/image.png; echo done'
    )
  })

  it('unwraps each supported URL quote style without changing other references', () => {
    const code = `first @url:'https://one.example'
second @url:"https://two.example" @file:\`src/app.ts\``

    expect(copyableCodeText(code)).toBe('first https://one.example\nsecond https://two.example @file:`src/app.ts`')
  })

  it('keeps hostless or malformed HTTP(S) URL directives raw', () => {
    for (const code of [
      '@url:http://',
      '@url:https://',
      '@url:http://?',
      '@url:https://?',
      '@url:http://:80',
      '@url:https://#fragment',
    ]) {
      expect(copyableCodeText(code)).toBe(code)
    }
  })

  it('leaves invalid URL directives and other reference kinds unchanged', () => {
    const code = '@url:ftp://example.com @url:example.com @image:https://example.com/image.png'

    expect(copyableCodeText(code)).toBe(code)
  })
})
