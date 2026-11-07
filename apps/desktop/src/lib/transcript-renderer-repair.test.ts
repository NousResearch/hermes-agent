import { strFromU8, unzipSync } from 'fflate'
import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderTranscript } from './transcript-render'

const url = 'https://example.org/a_(b)'
const args = JSON.stringify({ query: '[label](https://example.org/tool)' })

const doc: TranscriptDocument = {
  version: 1,
  session_id: 'session',
  title: 'Transcript',
  exported_at: '2026-01-01T00:00:00Z',
  options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_tools: true, include_reasoning: true, include_links: true },
  warnings: [],
  messages: [
    {
      role: 'assistant',
      speaker: 'Assistant',
      text: `Visit (${url}).`,
      reasoning: 'See https://example.org/reason',
      tool_calls: [{ name: 'lookup', arguments: args }]
    }
  ]
}

const unpack = (format: 'docx' | 'odt', value = doc) =>
  unzipSync(Uint8Array.from(atob(renderTranscript(value, format).content), c => c.charCodeAt(0)))

const parse = (source: string) => {
  const result = new DOMParser().parseFromString(source, 'application/xml')
  expect(result.querySelector('parsererror')).toBeNull()

  return result
}

it('preserves balanced targets across real DOCX, ODT, HTML and Markdown consumers', () => {
  const word = unpack('docx')
  const relationships = parse(strFromU8(word['word/_rels/document.xml.rels']))
  expect([...relationships.getElementsByTagName('Relationship')].map(x => x.getAttribute('Target'))).toContain(url)
  const odt = parse(strFromU8(unpack('odt')['content.xml']))
  expect(
    [...odt.getElementsByTagNameNS('urn:oasis:names:tc:opendocument:xmlns:text:1.0', 'a')].map(x =>
      x.getAttributeNS('http://www.w3.org/1999/xlink', 'href')
    )
  ).toContain(url)
  expect(odt.documentElement.textContent).toContain(`Visit (${url}).`)
  const html = new DOMParser().parseFromString(renderTranscript(doc, 'html').content, 'text/html')
  expect([...html.querySelectorAll('a')].map(a => a.getAttribute('href'))).toContain(url)
  expect(renderTranscript(doc, 'md').content).toContain(`](<${url}>)`)
})

it('links selected details and retains literal JSON in package and HTML output, with off switch', () => {
  for (const enabled of [true, false]) {
    const value = { ...doc, options: { ...doc.options, include_links: enabled } }
    const word = unpack('docx', value)
    const wordText = parse(strFromU8(word['word/document.xml']))
    const rels = parse(strFromU8(word['word/_rels/document.xml.rels']))

    const wordLinks = [...rels.getElementsByTagName('Relationship')]
      .filter(x => x.getAttribute('Type')?.endsWith('/hyperlink'))
      .map(x => x.getAttribute('Target'))

    const odt = parse(strFromU8(unpack('odt', value)['content.xml']))

    const odtLinks = [...odt.getElementsByTagNameNS('urn:oasis:names:tc:opendocument:xmlns:text:1.0', 'a')].map(x =>
      x.getAttributeNS('http://www.w3.org/1999/xlink', 'href')
    )

    const html = new DOMParser().parseFromString(renderTranscript(value, 'html').content, 'text/html')
    expect(wordText.documentElement.textContent).toContain(args)
    expect(odt.documentElement.textContent).toContain(args)
    expect(html.body.textContent).toContain(args)

    for (const links of [wordLinks, odtLinks, [...html.querySelectorAll('a')].map(a => a.getAttribute('href'))]) {
      if (enabled)
        {expect(links).toEqual(expect.arrayContaining(['https://example.org/tool', 'https://example.org/reason']))}
      else {expect(links).toHaveLength(0)}
    }

    const md = renderTranscript(value, 'md').content
    expect(md).toContain('https://example.org/tool')
    expect(md).toContain('https://example.org/reason')

    if (!enabled) {expect(md).not.toContain('](<https://')}
  }
})

it('retains literal tool arguments and results in HTML', () => {
  const literal = 'run [label](https://example.org/tool)'

  const value = {
    ...doc,
    messages: [
      { ...doc.messages[0], tool_calls: [{ name: 'run', arguments: literal }] },
      { role: 'tool', speaker: 'Tool', text: literal }
    ]
  }

  const html = new DOMParser().parseFromString(renderTranscript(value, 'html').content, 'text/html')
  expect([...html.querySelectorAll('details .body')].filter(body => body.textContent === literal)).toHaveLength(2)
})
