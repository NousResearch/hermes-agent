import { describe, expect, it } from 'vitest'

import { classifyCodeSpan, isBarePath, linkifyBarePaths, pathFromMarkdownHref, pathMarkdownHref, splitBarePath } from './path-refs'

describe('linkifyBarePaths', () => {
  it('rewrites a bare absolute path into a preview link, keeping the path as the label', () => {
    expect(linkifyBarePaths('Report: /Users/me/report.md')).toBe(
      'Report: [/Users/me/report.md](#path/%2FUsers%2Fme%2Freport.md)'
    )
  })

  it('handles ~/ and file:// paths', () => {
    expect(linkifyBarePaths('see ~/notes.md')).toBe('see [~/notes.md](#path/~%2Fnotes.md)')
    expect(linkifyBarePaths('see file:///srv/data/notes.txt')).toBe(
      'see [file:///srv/data/notes.txt](#path/file%3A%2F%2F%2Fsrv%2Fdata%2Fnotes.txt)'
    )
  })

  it('links media extensions through the same door (the renderer picks the player)', () => {
    expect(linkifyBarePaths('clip at /tmp/demo.mp4')).toBe('clip at [/tmp/demo.mp4](#path/%2Ftmp%2Fdemo.mp4)')
  })

  it('re-emits sentence punctuation after the link', () => {
    expect(linkifyBarePaths('Wrote /tmp/out.txt.')).toBe('Wrote [/tmp/out.txt](#path/%2Ftmp%2Fout.txt).')
    expect(linkifyBarePaths('Wrote /tmp/out.txt, then stopped')).toBe(
      'Wrote [/tmp/out.txt](#path/%2Ftmp%2Fout.txt), then stopped'
    )
  })

  it('does not double-wrap a path that is already a markdown link target', () => {
    const authored = '[report](/home/user/report.md)'

    expect(linkifyBarePaths(authored)).toBe(authored)
  })

  it('does not touch a path inside an autolink or a URL', () => {
    expect(linkifyBarePaths('<file:///x/y.md>')).toBe('<file:///x/y.md>')
    expect(linkifyBarePaths('https://example.com/docs/guide.md')).toBe('https://example.com/docs/guide.md')
  })

  it('leaves relative paths, directories and extension-less paths alone', () => {
    expect(linkifyBarePaths('run docs/guide.md')).toBe('run docs/guide.md')
    expect(linkifyBarePaths('in /usr/local/bin')).toBe('in /usr/local/bin')
    expect(linkifyBarePaths('ratio 3/4')).toBe('ratio 3/4')
    expect(linkifyBarePaths('/etc/hosts')).toBe('/etc/hosts')
  })

  it('leaves unknown extensions alone', () => {
    expect(linkifyBarePaths('/tmp/archive.tar.gz')).toBe('/tmp/archive.tar.gz')
    expect(linkifyBarePaths('/tmp/thing.xyz')).toBe('/tmp/thing.xyz')
  })

  it('stops at markdown-significant punctuation so a sentence cannot be swallowed', () => {
    expect(linkifyBarePaths('(see /tmp/a.md)')).toBe('(see [/tmp/a.md](#path/%2Ftmp%2Fa.md))')
    expect(linkifyBarePaths('"/tmp/a.md" is it')).toBe('"[/tmp/a.md](#path/%2Ftmp%2Fa.md)" is it')
  })

  it('does not extend a path through markdown brackets', () => {
    // Brackets are excluded from path characters so `[see /tmp/a.md]` cannot
    // swallow the closing bracket; a path containing them is left alone.
    expect(linkifyBarePaths('[see /tmp/a.md]')).toBe('[see [/tmp/a.md](#path/%2Ftmp%2Fa.md)]')
    expect(linkifyBarePaths('/tmp/[draft].md')).toBe('/tmp/[draft].md')
  })

  it('links every path in a line', () => {
    const out = linkifyBarePaths('/a/one.md and /b/two.txt')

    expect(out.match(/#path\//g)).toHaveLength(2)
  })

  it('is a no-op on text without a slash', () => {
    expect(linkifyBarePaths('nothing here')).toBe('nothing here')
  })
})

describe('splitBarePath', () => {
  it('splits trailing punctuation from the path', () => {
    expect(splitBarePath('/tmp/a.md.')).toEqual({ trailing: '.', value: '/tmp/a.md' })
    expect(splitBarePath('/tmp/a.md')).toEqual({ trailing: '', value: '/tmp/a.md' })
  })
})

describe('isBarePath', () => {
  it('accepts the exact shapes linkifyBarePaths rewrites', () => {
    expect(isBarePath('/tmp/a.md')).toBe(true)
    expect(isBarePath('~/a.md')).toBe(true)
    expect(isBarePath('file:///tmp/a.md')).toBe(true)
  })

  it('rejects anything with surrounding text or no known extension', () => {
    expect(isBarePath('see /tmp/a.md')).toBe(false)
    expect(isBarePath('/tmp/a')).toBe(false)
    expect(isBarePath('docs/a.md')).toBe(false)
  })
})

describe('classifyCodeSpan', () => {
  it('classifies a URL that fills the span', () => {
    expect(classifyCodeSpan('http://localhost:8931/')).toBe('url')
    expect(classifyCodeSpan('https://example.com/a?b=1#c')).toBe('url')
  })

  it('classifies absolute, home and file: paths — any extension, since a code span is deliberate', () => {
    expect(classifyCodeSpan('/Users/me/dev/reports/register.md')).toBe('path')
    expect(classifyCodeSpan('~/todo.txt')).toBe('path')
    expect(classifyCodeSpan('file:///srv/x.html')).toBe('path')
    expect(classifyCodeSpan('/etc/hosts')).toBe('path')
  })

  it('rejects commands, symbols, relative paths and multi-token spans', () => {
    expect(classifyCodeSpan('git checkout --')).toBeNull()
    expect(classifyCodeSpan('npm run dev')).toBeNull()
    expect(classifyCodeSpan('useEffect')).toBeNull()
    expect(classifyCodeSpan('docs/guide.md')).toBeNull()
    expect(classifyCodeSpan('/tmp/a.md and /tmp/b.md')).toBeNull()
    expect(classifyCodeSpan('view-source:http://x/')).toBeNull()
    expect(classifyCodeSpan('')).toBeNull()
  })
})

describe('path href door', () => {
  it('round-trips through #path/', () => {
    expect(pathFromMarkdownHref(pathMarkdownHref('/tmp/a b.md'))).toBe('/tmp/a b.md')
    expect(pathFromMarkdownHref('#preview/%2Ftmp')).toBeNull()
    expect(pathFromMarkdownHref(undefined)).toBeNull()
  })
})
