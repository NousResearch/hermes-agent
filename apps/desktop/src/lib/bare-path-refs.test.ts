import { describe, expect, it } from 'vitest'

import { classifyCodeSpan, linkifyBarePaths, pathFromMarkdownHref, pathMarkdownHref } from './bare-path-refs'
import { preprocessMarkdown } from './markdown-preprocess'

describe('linkifyBarePaths', () => {
  it('rewrites a bare absolute path into a #path/ link labelled with the path', () => {
    expect(linkifyBarePaths('Report: /Users/me/report.md')).toBe(
      'Report: [/Users/me/report.md](#path/%2FUsers%2Fme%2Freport.md)'
    )
  })

  it('handles ~/ and file:// paths, encoding the tilde in the href', () => {
    expect(linkifyBarePaths('see ~/notes.md')).toBe('see [~/notes.md](#path/%7E%2Fnotes.md)')
    expect(linkifyBarePaths('see file:///srv/data/notes.txt')).toBe(
      'see [file:///srv/data/notes.txt](#path/file%3A%2F%2F%2Fsrv%2Fdata%2Fnotes.txt)'
    )
  })

  it('keeps sentence punctuation and wrapping brackets outside the link', () => {
    expect(linkifyBarePaths('Wrote /tmp/out.txt.')).toBe('Wrote [/tmp/out.txt](#path/%2Ftmp%2Fout.txt).')
    expect(linkifyBarePaths('(see /tmp/a.md)')).toBe('(see [/tmp/a.md](#path/%2Ftmp%2Fa.md))')
    expect(linkifyBarePaths('"/tmp/a.md" is it')).toBe('"[/tmp/a.md](#path/%2Ftmp%2Fa.md)" is it')
  })

  it('links every path in a line', () => {
    expect(linkifyBarePaths('/a/one.md and /b/two.txt').match(/#path\//g)).toHaveLength(2)
  })

  it('leaves relative paths, directories, ratios and unknown extensions alone', () => {
    for (const text of [
      'run docs/guide.md',
      'run ./guide.md or ../guide.md',
      'in /usr/local/bin',
      'ratio 3/4',
      '/etc/hosts',
      '/tmp/archive.tar.gz',
      '/tmp/a.md.bak',
      'nothing here'
    ]) {
      expect(linkifyBarePaths(text)).toBe(text)
    }
  })

  it('never links executables or installers', () => {
    for (const text of ['/Applications/Evil.app', '/tmp/setup.exe', '~/Downloads/x.dmg', '/tmp/run.command']) {
      expect(linkifyBarePaths(text)).toBe(text)
    }
  })

  it('does not touch paths inside URLs, autolinks or HTML attributes', () => {
    expect(linkifyBarePaths('<file:///x/y.md>')).toBe('<file:///x/y.md>')
    expect(linkifyBarePaths('<https://x.dev/?f=/tmp/a.md>')).toBe('<https://x.dev/?f=/tmp/a.md>')
    expect(linkifyBarePaths('http://localhost:8000/docs/a.md')).toBe('http://localhost:8000/docs/a.md')
    expect(linkifyBarePaths('<a href="/tmp/a.md">x</a>')).toBe('<a href="/tmp/a.md">x</a>')
  })
})

describe('preprocessMarkdown with bare paths', () => {
  it('links a prose path but not an authored link, its label, a code span or a fence', () => {
    const out = preprocessMarkdown(
      'See /tmp/a.md, [report](/home/u/r.md), [see /tmp/b.md](https://x.dev) and `/tmp/c.md`\n\n```sh\ncat /tmp/d.md\n```'
    )

    expect(out).toContain('[/tmp/a.md](#path/%2Ftmp%2Fa.md),')
    expect(out).toContain('[report](#preview/%2Fhome%2Fu%2Fr.md)')
    expect(out).toContain('[see /tmp/b.md](https://x.dev)')
    expect(out).toContain('`/tmp/c.md`')
    expect(out).toContain('cat /tmp/d.md')
    expect(out.match(/#path\//g)).toHaveLength(1)
  })

  it('does not link a path inside a bare URL', () => {
    expect(preprocessMarkdown('open https://x.dev/?f=/tmp/a.md now')).toBe('open <https://x.dev/?f=/tmp/a.md> now')
  })
})

describe('classifyCodeSpan', () => {
  it('classifies a URL that fills the span', () => {
    expect(classifyCodeSpan('http://localhost:8931/')).toBe('url')
    expect(classifyCodeSpan('https://example.com/a?b=1#c')).toBe('url')
  })

  it('classifies absolute, home and file:// paths with any extension', () => {
    expect(classifyCodeSpan('/Users/me/dev/reports/register.md')).toBe('path')
    expect(classifyCodeSpan('~/todo.txt')).toBe('path')
    expect(classifyCodeSpan('file:///srv/x.html')).toBe('path')
    expect(classifyCodeSpan('/etc/hosts')).toBe('path')
  })

  it('rejects commands, symbols, relative paths, executables and multi-token spans', () => {
    for (const text of [
      'git checkout --',
      'useEffect',
      'docs/guide.md',
      '/',
      '/tmp/a.md and /tmp/b.md',
      'view-source:http://x/',
      'javascript:alert(1)',
      '/Applications/Evil.app',
      '/Applications/Evil.app/',
      'C:\\Tools\\setup.exe',
      '~/bin/install.command',
      ''
    ]) {
      expect(classifyCodeSpan(text)).toBeNull()
    }
  })
})

describe('#path/ href', () => {
  it('round-trips and ignores other doors', () => {
    expect(pathFromMarkdownHref(pathMarkdownHref('/tmp/a b.md'))).toBe('/tmp/a b.md')
    expect(pathFromMarkdownHref(pathMarkdownHref('~/a.md'))).toBe('~/a.md')
    expect(pathFromMarkdownHref('#preview/%2Ftmp')).toBeNull()
    expect(pathFromMarkdownHref('#path/%E0%A4%A')).toBeNull()
    expect(pathFromMarkdownHref(undefined)).toBeNull()
  })
})
