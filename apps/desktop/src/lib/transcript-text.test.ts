import { strFromU8, unzipSync } from 'fflate'
import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderTranscript } from './transcript-render'
import { safeTranscriptLink, transcriptTextParts } from './transcript-text'

it('does not activate ambiguous whitespace or control characters in a web address', () => {
  for (const char of ['\u00a0', '\u2000', '\u2028', '\u0000', '\n']) {
    expect(safeTranscriptLink(`https://example.com/a${char}b`)).toBe(false)
  }
})

it('preserves the exact original wording when clickable links are off', () => {
  const text = 'Read [the brief](https://example.com/brief?a=1&b=2).'
  expect(transcriptTextParts(text, false)).toEqual([{ text }])
  expect(transcriptTextParts(text, true)).toEqual([
    { text: 'Read ' },
    { text: 'the brief', href: 'https://example.com/brief?a=1&b=2' },
    { text: '.' }
  ])
})

it('keeps balanced parentheses in bare URLs and trims prose wrappers', () => {
  const source = '(https://example.org/a_(b)).'
  expect(transcriptTextParts(source, true)).toEqual([
    { text: '(' },
    { text: 'https://example.org/a_(b)', href: 'https://example.org/a_(b)' },
    { text: ').' }
  ])
})

it('keeps adjacent named links separate with or without punctuation', () => {
  for (const between of ['', ',']) {
    const source = `[a](https://one.example)${between}[b](https://two.example)`
    expect(transcriptTextParts(source, true)).toEqual([
      { text: 'a', href: 'https://one.example' },
      ...(between ? [{ text: between }] : []),
      { text: 'b', href: 'https://two.example' }
    ])
  }
})


it('balances nested named destinations without consuming following prose or links', () => {
  const source = '[a](https://one.example/a_(b_(c))),next[b](https://two.example)'
  expect(transcriptTextParts(source, true)).toEqual([
    { text: 'a', href: 'https://one.example/a_(b_(c))' },
    { text: ',next' },
    { text: 'b', href: 'https://two.example' }
  ])
  expect(transcriptTextParts(source, false)).toEqual([{ text: source }])
})

it('retains unsafe and malformed named source without swallowing a later safe link', () => {
  for (const prefix of ['[bad](https://person:password@bad.example)', '[bad](https://bad.example ']) {
    const parts = transcriptTextParts(`${prefix}[good](https://two.example)`, true)
    expect(parts).toEqual([{ text: prefix }, { text: 'good', href: 'https://two.example' }])
  }
})

it('never replaces named-link syntax inside literal or structured tool text', () => {
  const source = '[a](https://one.example),[b](https://two.example)'

  for (const literal of [source, JSON.stringify({ query: source, limit: 5 }), JSON.stringify([source])]) {
    const parts = transcriptTextParts(literal, true, literal === source)
    expect(parts.map(part => part.text).join('')).toBe(literal)
    expect(parts.filter(part => part.href).map(part => part.href)).toEqual([
      'https://one.example',
      'https://two.example'
    ])
  }
})

it('retains overlong destinations without activating them or losing the next link', () => {
  const long = `[long](https://one.example/${'x'.repeat(5000)})`
  expect(transcriptTextParts(`${long} [good](https://two.example)`, true)).toEqual([
    { text: `${long} ` },
    { text: 'good', href: 'https://two.example' }
  ])
})

it('preserves case-insensitive web schemes when linking adjacent labels or bare URLs', () => {
  expect(transcriptTextParts('[a](HTTPS://one.example)[b](HTTP://two.example)', true)).toEqual([
    { text: 'a', href: 'HTTPS://one.example' },
    { text: 'b', href: 'HTTP://two.example' }
  ])
  expect(transcriptTextParts('HTTPS://one.example', true)).toEqual([
    { text: 'HTTPS://one.example', href: 'HTTPS://one.example' }
  ])
})

it('retains non-web protocol wrappers without activating a nested web-looking substring', () => {
  for (const prefix of ['javascript:', 'file:///', 'not', 'name@', 'prefix.']) {
    const source = `${prefix}https://safe.example/path`
    expect(transcriptTextParts(source, true)).toEqual([{ text: source }])
  }

  expect(transcriptTextParts('See https://safe.example/path', true)).toEqual([
    { text: 'See ' },
    { text: 'https://safe.example/path', href: 'https://safe.example/path' }
  ])
})

it.each(
  (['docx', 'odt', 'html', 'md'] as const).flatMap(format =>
    ['', '/a_(b)'].map(path => ({ format, path }))
  )
)(
  'preserves adjacent targets, link-off source and literal arguments in the real $format consumer ($path)',
  ({ format, path }) => {
    const one = `https://one.example${path}`
    const two = 'https://two.example'
    const source = `[a](${one}),[b](${two})`
    const args = JSON.stringify({ query: source, limit: 5 })

    const xml = (value: Uint8Array) => {
      const parsed = new DOMParser().parseFromString(strFromU8(value), 'application/xml')
      expect(parsed.querySelector('parsererror')).toBeNull()

      return parsed
    }

    for (const enabled of [true, false]) {
      const doc: TranscriptDocument = {
        version: 1,
        session_id: 'adjacent-sample',
        title: 'Adjacent links',
        exported_at: '2026-01-01T00:00:00Z',
        options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_tools: true, include_links: enabled },
        warnings: [],
        messages: [{ role: 'assistant', speaker: 'Assistant', text: source, tool_calls: [{ name: 'lookup', arguments: args }] }]
      }

      const rendered = renderTranscript(doc, format).content
      let links: (string | null)[] = []
      let body: string | null

      if (format === 'docx' || format === 'odt') {
        const archive = unzipSync(Uint8Array.from(atob(rendered), c => c.charCodeAt(0)))
        const content = xml(archive[format === 'docx' ? 'word/document.xml' : 'content.xml'])
        body = content.documentElement.textContent
        links = format === 'docx'
          ? [...xml(archive['word/_rels/document.xml.rels']).getElementsByTagName('Relationship')]
            .filter(link => link.getAttribute('Type')?.endsWith('/hyperlink'))
            .map(link => link.getAttribute('Target'))
          : [...content.getElementsByTagNameNS('urn:oasis:names:tc:opendocument:xmlns:text:1.0', 'a')]
            .map(link => link.getAttributeNS('http://www.w3.org/1999/xlink', 'href'))
      } else {
        const parsed = new DOMParser().parseFromString(rendered, 'text/html')
        body = parsed.body.textContent
        links = [...parsed.querySelectorAll('a')].map(link => link.getAttribute('href'))
      }

      expect(body).toContain(args)

      if (format === 'md') {
        if (enabled) {
          expect(rendered).toContain(`[a](<${one}>),[b](<${two}>)`)
        } else {
          expect(rendered).not.toContain('](<https://')
          expect(links).toHaveLength(0)
          expect(body).toContain(source)
        }
      } else {
        expect(body).toContain(enabled ? 'a,b' : source)

        if (enabled) {
          expect(links, `${format}: ${JSON.stringify(links)}`).toEqual(expect.arrayContaining([one, two]))
        } else {
          expect(links).toHaveLength(0)
        }
      }
    }
  }
)
