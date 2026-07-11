import { describe, expect, it } from 'vitest'

import {
  extractAutoPreviewTargets,
  extractLocalMarkdownPreviewTargets,
  extractPreviewTargets,
  previewTargetFromMarkdownHref,
  stripPreviewTargets
} from './preview-targets'

describe('preview target detection', () => {
  it('does not infer preview targets from raw paths or URLs', () => {
    expect(extractPreviewTargets('Preview: http://localhost:5173/')).toEqual([])
    expect(extractPreviewTargets('Open index.html\n/tmp/demo.html\nhttp://localhost:5173/')).toEqual([])
  })

  it('decodes preview markdown hrefs', () => {
    expect(previewTargetFromMarkdownHref('#preview/%2Ftmp%2Fdemo.html')).toBe('/tmp/demo.html')
    expect(previewTargetFromMarkdownHref('#preview:%2Ftmp%2Fdemo.html')).toBe('/tmp/demo.html')
    expect(previewTargetFromMarkdownHref('#media:%2Ftmp%2Fdemo.mp4')).toBeNull()
  })

  it('extracts preview targets from already-rendered preview markers', () => {
    expect(extractPreviewTargets('[Preview: demo.html](#preview:%2Ftmp%2Fdemo.html)')).toEqual(['/tmp/demo.html'])
  })

  it('strips preview targets from visible assistant text', () => {
    expect(stripPreviewTargets('ready\n/tmp/mycelium-bunnies.html\nopen it')).toBe(
      'ready\n/tmp/mycelium-bunnies.html\nopen it'
    )
    expect(stripPreviewTargets('[Preview: demo.html](#preview:%2Ftmp%2Fdemo.html)\nopen it')).toBe('open it')
  })
})

describe('normal Markdown local preview links', () => {
  it('extracts an absolute local link for footer rendering', () => {
    expect(extractLocalMarkdownPreviewTargets('[artifact](/tmp/report.md)', false)).toEqual(['/tmp/report.md'])
  })

  it('requires relative mode for normal relative Markdown links', () => {
    const source = '[artifact](./report.md)'

    expect(extractLocalMarkdownPreviewTargets(source, false)).toEqual([])
    expect(extractLocalMarkdownPreviewTargets(source, true)).toEqual(['./report.md'])
  })

  it.each([
    '`[artifact](/tmp/inline.md)`',
    '```markdown\n[artifact](/tmp/fenced.md)\n```',
    '![chart](/tmp/chart.png)',
    '--- [old](/tmp/old.md)',
    '2026-01-01T12:00:00Z INFO [report](/tmp/logged.md)'
  ])('does not extract ignored Markdown content: %s', source => {
    expect(extractLocalMarkdownPreviewTargets(source, false)).toEqual([])
  })
})

describe('automatic local preview target detection', () => {
  it('extracts absolute macOS and Linux paths', () => {
    expect(
      extractAutoPreviewTargets(
        'Saved /Users/andrew/My Project/report.md and /tmp/hermes/output.pdf for review.'
      )
    ).toEqual(['/Users/andrew/My Project/report.md', '/tmp/hermes/output.pdf'])
  })

  it('extracts home-relative paths and file URLs', () => {
    expect(extractAutoPreviewTargets('Open ~/Documents/report.md or file:///Users/andrew/demo.html.')).toEqual([
      '~/Documents/report.md',
      'file:///Users/andrew/demo.html'
    ])
  })

  it('extracts Windows drive paths with backslash and slash separators', () => {
    expect(
      extractAutoPreviewTargets(
        'Saved C:\\Users\\Andrew\\My Documents\\report.pdf and D:/Exports/data.csv.'
      )
    ).toEqual(['C:\\Users\\Andrew\\My Documents\\report.pdf', 'D:/Exports/data.csv'])
  })

  it('extracts parenthesized POSIX paths and localhost file URLs', () => {
    expect(
      extractAutoPreviewTargets('Saved /tmp/report(1).pdf and file://localhost/tmp/transcript.txt.')
    ).toEqual(['/tmp/report(1).pdf', 'file://localhost/tmp/transcript.txt'])
  })

  it('deduplicates while preserving first-seen order', () => {
    expect(extractAutoPreviewTargets('/tmp/a.md then /tmp/b.pdf then /tmp/a.md')).toEqual([
      '/tmp/a.md',
      '/tmp/b.pdf'
    ])
  })

  it('strips trailing prose punctuation', () => {
    expect(extractAutoPreviewTargets('Open (/tmp/report.md), then "/tmp/chart.png".')).toEqual([
      '/tmp/report.md',
      '/tmp/chart.png'
    ])
  })

  it('ignores paths inside fenced, inline, and indented code', () => {
    expect(
      extractAutoPreviewTargets(
        '```text\n/tmp/fenced.md\n```\nInline `/tmp/inline.md`\n    /tmp/indented.md\nOpen /tmp/real.md'
      )
    ).toEqual(['/tmp/real.md'])
  })

  it('ignores complete diff hunks, log lines, and stack traces', () => {
    expect(
      extractAutoPreviewTargets(
        [
          'diff --git a/report.md b/report.md',
          'index 1111111..2222222 100644',
          '--- /tmp/old.md',
          '+++ /tmp/new.md',
          '@@ -1 +1 @@',
          '-Saved /tmp/removed.md',
          '+Saved /tmp/added.md',
          ' unchanged /tmp/context.md',
          'diff --git a/old.md b/new.md',
          'similarity index 100%',
          'rename from /tmp/old.md',
          'rename to /tmp/new.md',
          'diff --git a/a.png b/b.png',
          'Binary files /tmp/a.png and /tmp/b.png differ',
          '2026-01-01T12:00:00Z wrote /tmp/timestamped.md',
          '[12:00:00] wrote /tmp/bracketed.md',
          '12:34:56 INFO wrote /tmp/clock.md',
          'Jul 10 12:34:56 host app[1]: wrote /tmp/syslog.md',
          'RuntimeError: failed reading /tmp/python-error.md',
          'java.lang.RuntimeException: failed reading /tmp/java-error.md',
          '    at load (/tmp/stack.ts:10:2)',
          'Open /tmp/real.md'
        ].join('\n')
      )
    ).toEqual(['/tmp/real.md'])
  })

  it('ignores local paths used as inline and reference-style Markdown image destinations', () => {
    expect(
      extractAutoPreviewTargets(
        ['![inline](/tmp/inline.png)', '![chart][img]', '[img]: /tmp/reference.png', '![shortcut]', '[shortcut]: /tmp/shortcut.png'].join('\n')
      )
    ).toEqual([])
  })

  it('does not infer ordinary local URLs', () => {
    expect(extractAutoPreviewTargets('Visit http://localhost:5173 or https://example.com/report.pdf')).toEqual([])
  })

  it('ignores domains and secret-looking paths', () => {
    expect(
      extractAutoPreviewTargets(
        [
          'Docs example.com/report.md;',
          'skip /tmp/.env, /tmp/api-key.txt, /tmp/client-private-key.pdf,',
          '/tmp/certificate.pem, and /Users/andrew/.ssh/config.pdf.'
        ].join(' ')
      )
    ).toEqual([])
  })

  it('extracts common text and data artifacts supported by the preview rail', () => {
    expect(extractAutoPreviewTargets('Saved /tmp/report.csv /tmp/transcript.txt /tmp/data.json /tmp/config.yaml')).toEqual(
      ['/tmp/report.csv', '/tmp/transcript.txt', '/tmp/data.json']
    )
  })

  it('does not treat prose ending in a supported extension as a path', () => {
    expect(extractAutoPreviewTargets('Please document the changes in report.md before review.')).toEqual([])
  })

  it('ignores unsupported and suffixed file extensions', () => {
    expect(
      extractAutoPreviewTargets('Skip /tmp/report.docx, /tmp/report.md.bak, and file:///tmp/page.html#draft.')
    ).toEqual([])
  })

  it('ignores invalid file URLs', () => {
    expect(extractAutoPreviewTargets('Broken file:///%E0%A4%A.md but valid /tmp/report.md')).toEqual([
      '/tmp/report.md'
    ])
  })

  it('caps the default output at three targets', () => {
    expect(extractAutoPreviewTargets('/tmp/a.md /tmp/b.md /tmp/c.md /tmp/d.md')).toEqual([
      '/tmp/a.md',
      '/tmp/b.md',
      '/tmp/c.md'
    ])
  })

  it('normalizes fractional and non-finite maxTargets values', () => {
    expect(extractAutoPreviewTargets('/tmp/a.md /tmp/b.md', { maxTargets: 1.5 })).toEqual(['/tmp/a.md'])
    expect(extractAutoPreviewTargets('/tmp/a.md /tmp/b.md', { maxTargets: Number.NaN })).toEqual([
      '/tmp/a.md',
      '/tmp/b.md'
    ])
  })

  it('does not extract relative paths by default', () => {
    expect(extractAutoPreviewTargets('Open ./artifacts/report.md and ../shared/chart.png')).toEqual([])
  })

  it('extracts relative paths only when explicitly enabled', () => {
    expect(
      extractAutoPreviewTargets('Open ./artifacts/report.md and ../shared/chart.png', {
        includeRelative: true
      })
    ).toEqual(['./artifacts/report.md', '../shared/chart.png'])
  })
})
