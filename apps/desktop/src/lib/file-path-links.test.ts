import { describe, expect, it } from 'vitest'

import { filePathFromMarkdownHref, isLoneFilePath, linkifyFilePaths, splitFilePathSuffix } from './file-path-links'
import { preprocessMarkdown } from './markdown-preprocess'

function hrefIn(markdown: string): null | string {
  const match = markdown.match(/\(#file\/[^)]+\)/)

  return match ? match[0].slice(1, -1) : null
}

describe('linkifyFilePaths', () => {
  it('linkifies an absolute POSIX path', () => {
    const out = linkifyFilePaths('saved to /Users/me/out.ts done')

    expect(out).toContain('[/Users/me/out.ts](#file/')
    expect(filePathFromMarkdownHref(hrefIn(out))).toBe('/Users/me/out.ts')
  })

  it('linkifies a ~/ home path', () => {
    const out = linkifyFilePaths('see ~/notes/todo.md please')

    expect(filePathFromMarkdownHref(hrefIn(out))).toBe('~/notes/todo.md')
  })

  it('keeps a :line:col suffix in the visible label', () => {
    const out = linkifyFilePaths('error at /a/b.ts:42:7 now')

    expect(out).toContain('[/a/b.ts:42:7](#file/')
    expect(filePathFromMarkdownHref(hrefIn(out))).toBe('/a/b.ts:42:7')
  })

  it('linkifies a Windows drive path', () => {
    const out = linkifyFilePaths('wrote C:\\Users\\me\\out.txt ok')

    expect(out).toContain('](#file/')
    expect(filePathFromMarkdownHref(hrefIn(out))).toBe('C:\\Users\\me\\out.txt')
  })

  it('does not linkify relative paths', () => {
    expect(linkifyFilePaths('edit src/app/x.tsx now')).toBe('edit src/app/x.tsx now')
  })

  it('does not linkify directories or extensionless paths', () => {
    expect(linkifyFilePaths('cd /usr/local/bin here')).toBe('cd /usr/local/bin here')
  })

  it('does not linkify domain paths (no scheme)', () => {
    expect(linkifyFilePaths('visit example.com/docs/intro.html now')).toBe('visit example.com/docs/intro.html now')
  })

  it('does not rewrite a path already inside a markdown link', () => {
    const input = '[label](/a/b.png)'

    expect(linkifyFilePaths(input)).toBe(input)
  })

  it('does not rewrite a path inside an autolink', () => {
    const input = '<https://example.com/a/b.ts>'

    expect(linkifyFilePaths(input)).toBe(input)
  })

  it('encodes parentheses so the markdown target stays intact', () => {
    const out = linkifyFilePaths('see /tmp/build(1)/out.txt now')

    expect(filePathFromMarkdownHref(hrefIn(out))).toBe('/tmp/build(1)/out.txt')
  })
})

describe('linkifyFilePaths — review-hardening regressions', () => {
  it('does not corrupt an existing markdown link whose target contains parentheses', () => {
    const input = 'See [the report](/Users/me/build(1)/out/result.json) for details.'

    expect(linkifyFilePaths(input)).toBe(input)
  })

  it('does not corrupt a #preview link target containing parentheses', () => {
    const input = '[r](#preview/path(x)/index.html)'

    expect(linkifyFilePaths(input)).toBe(input)
  })

  it('linkifies a long (>8 char) extension in full, not truncated', () => {
    const out = linkifyFilePaths('the file /home/user/data.properties now')

    expect(filePathFromMarkdownHref(hrefIn(out))).toBe('/home/user/data.properties')
    expect(out).not.toContain('data.properti]') // no truncated link
    expect(out).not.toContain(')es ') // no stray tail leaked as prose
  })

  it('handles iOS extensions (.storyboard, .entitlements) without truncation', () => {
    expect(filePathFromMarkdownHref(hrefIn(linkifyFilePaths('open /a/Main.storyboard now')))).toBe('/a/Main.storyboard')
    expect(filePathFromMarkdownHref(hrefIn(linkifyFilePaths('see /a/App.entitlements here')))).toBe(
      '/a/App.entitlements'
    )
  })

  it('does not truncate/corrupt a path with a hyphen after the extension', () => {
    // `/a/b.c-d` is ambiguous; the safe outcome is to leave it untouched rather
    // than emit a link to the truncated `/a/b.c`.
    const input = 'value /a/b.c-d here'

    expect(linkifyFilePaths(input)).toBe(input)
  })

  it('peels a trailing sentence period off the path', () => {
    const out = linkifyFilePaths('saved to /Users/me/out.ts.')

    expect(filePathFromMarkdownHref(hrefIn(out))).toBe('/Users/me/out.ts')
    expect(out).toContain('](#file/')
    expect(out.endsWith('.')).toBe(true) // the period stays as prose
  })
})

describe('splitFilePathSuffix', () => {
  it('strips :line:col for the open target while keeping the display', () => {
    expect(splitFilePathSuffix('/a/b.ts:42:7')).toEqual({ display: '/a/b.ts:42:7', path: '/a/b.ts' })
  })

  it('leaves a plain path untouched', () => {
    expect(splitFilePathSuffix('/a/b.ts')).toEqual({ display: '/a/b.ts', path: '/a/b.ts' })
  })
})

describe('preprocessMarkdown file-path linkification', () => {
  it('linkifies a path in prose but not inside a fenced code block', () => {
    const out = preprocessMarkdown('open /Users/me/a.txt\n\n```bash\ncat /Users/me/b.txt\n```\n')

    expect(out).toContain('[/Users/me/a.txt](#file/')
    expect(out).toContain('/Users/me/b.txt')
    expect(out).not.toContain('[/Users/me/b.txt](#file/')
  })

  it('linkifies a lone path inside inline code (follow-up #1)', () => {
    const out = preprocessMarkdown('run `/Users/me/c.txt` please')

    expect(out).toContain('[/Users/me/c.txt](#file/')
  })

  it('does not linkify inline code that is a command, not a lone path', () => {
    const out = preprocessMarkdown('run `cat /Users/me/c.txt` please')

    expect(out).not.toContain('#file/')
  })
})

describe('code-span path linkification (follow-up #1)', () => {
  it('isLoneFilePath recognizes exactly-one-path strings', () => {
    expect(isLoneFilePath('/Users/me/out.ts')).toBe(true)
    expect(isLoneFilePath('/Users/me/out.ts:42:7')).toBe(true)
    expect(isLoneFilePath('~/notes/todo.md')).toBe(true)
    expect(isLoneFilePath('C:\\Users\\me\\out.txt')).toBe(true)
    expect(isLoneFilePath('  /a/b.json  ')).toBe(true)
  })

  it('isLoneFilePath rejects commands, prose, and extensionless/relative paths', () => {
    expect(isLoneFilePath('cat /a/b.ts')).toBe(false)
    expect(isLoneFilePath('/a/b.ts and more')).toBe(false)
    expect(isLoneFilePath('/usr/local/bin')).toBe(false)
    expect(isLoneFilePath('src/app/x.tsx')).toBe(false)
  })

  it('linkifies a path emitted as a one-line code block', () => {
    const out = preprocessMarkdown('Here it is:\n\n```\n/Users/me/out.ts\n```\n')

    expect(out).toContain('[/Users/me/out.ts](#file/')
    expect(out).not.toContain('```')
  })

  it('linkifies a one-line code block that carries a language tag', () => {
    const out = preprocessMarkdown('```text\n/Users/me/report.pdf\n```')

    expect(out).toContain('[/Users/me/report.pdf](#file/')
  })

  it('does NOT linkify paths inside a multi-line code block', () => {
    const out = preprocessMarkdown('```bash\ncd /Users/me\ncat /Users/me/b.txt\n```')

    expect(out).not.toContain('#file/')
    expect(out).toContain('/Users/me/b.txt')
  })
})
