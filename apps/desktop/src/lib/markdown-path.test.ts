import { describe, expect, it } from 'vitest'

import { stripMarkdownPathWrappers } from '@hermes/shared/markdown-path'

describe('stripMarkdownPathWrappers', () => {
  it('unwraps bold markers around file names (#95713)', () => {
    expect(stripMarkdownPathWrappers('**Open xyz.pdf**')).toBe('Open xyz.pdf')
    expect(stripMarkdownPathWrappers('**xyz.pdf**')).toBe('xyz.pdf')
    expect(stripMarkdownPathWrappers('**/home/user/xyz.pdf**')).toBe('/home/user/xyz.pdf')
    expect(stripMarkdownPathWrappers('**C:\\Users\\me\\xyz.pdf**')).toBe('C:\\Users\\me\\xyz.pdf')
  })

  it('unwraps underscore emphasis around labels', () => {
    expect(stripMarkdownPathWrappers('__Open my report__')).toBe('Open my report')
    expect(stripMarkdownPathWrappers('__quarterly report.pdf__')).toBe('quarterly report.pdf')
  })

  it('leaves dunder-style names alone', () => {
    expect(stripMarkdownPathWrappers('__init__')).toBe('__init__')
    expect(stripMarkdownPathWrappers('__pycache__')).toBe('__pycache__')
    expect(stripMarkdownPathWrappers('__main__')).toBe('__main__')
    expect(stripMarkdownPathWrappers('src/__init__.py')).toBe('src/__init__.py')
  })

  it('strips @url: directive prefixes', () => {
    expect(stripMarkdownPathWrappers('@url:xyz.pdf')).toBe('xyz.pdf')
    expect(stripMarkdownPathWrappers('@url: https://example.com')).toBe('https://example.com')
    expect(stripMarkdownPathWrappers('**@url:xyz.pdf**')).toBe('xyz.pdf')
  })

  it('unwraps backtick-quoted paths', () => {
    expect(stripMarkdownPathWrappers('`~/notes/todo.md`')).toBe('~/notes/todo.md')
  })

  it('handles single-asterisk emphasis and stray unmatched markers', () => {
    expect(stripMarkdownPathWrappers('*report.pdf*')).toBe('report.pdf')
    // Matches the exact Windows error from #95713: trailing '**' on the path.
    expect(stripMarkdownPathWrappers('xyz.pdf**')).toBe('xyz.pdf')
    expect(stripMarkdownPathWrappers('`~/notes/todo.md')).toBe('~/notes/todo.md')
  })

  it('never touches interior characters or plain paths', () => {
    expect(stripMarkdownPathWrappers('/tmp/a*b/c.pdf')).toBe('/tmp/a*b/c.pdf')
    expect(stripMarkdownPathWrappers('/home/u/notes.md')).toBe('/home/u/notes.md')
    expect(stripMarkdownPathWrappers('  /home/u/notes.md  ')).toBe('/home/u/notes.md')
    expect(stripMarkdownPathWrappers('file:///C:/dir/report final.pdf')).toBe('file:///C:/dir/report final.pdf')
  })

  it('returns the original when everything would strip away', () => {
    expect(stripMarkdownPathWrappers('**')).toBe('**')
    expect(stripMarkdownPathWrappers('')).toBe('')
  })
})
