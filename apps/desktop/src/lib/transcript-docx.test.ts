import { strFromU8, unzipSync } from 'fflate'
import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderTranscript } from './transcript-render'

const image =
  'data:image/png;base64,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'

const doc: TranscriptDocument = {
  version: 1,
  session_id: 'sample-session',
  title: 'Planning session',
  exported_at: '2026-11-03T18:00:00Z',
  options: { ...DEFAULT_TRANSCRIPT_OPTIONS },
  warnings: [],
  messages: [
    {
      role: 'user',
      speaker: 'User',
      text: 'Please review this diagram.',
      timestamp: '2026-11-03T17:58:00Z',
      attachments: [{ kind: 'image', name: 'Project diagram', data_url: image, mime_type: 'image/png' }]
    }
  ]
}

function parts(value = doc) {
  return unzipSync(Uint8Array.from(atob(renderTranscript(value, 'docx').content), c => c.charCodeAt(0)))
}

it('routes OpenDocument output through the image-capable package writer', () => {
  const result = renderTranscript(doc, 'odt')
  const zip = unzipSync(Uint8Array.from(atob(result.content), c => c.charCodeAt(0)))
  expect(Object.keys(zip).filter(name => name.startsWith('Pictures/'))).toHaveLength(1)
})

function xml(text: string) {
  const parsed = new DOMParser().parseFromString(text, 'application/xml')
  expect(parsed.querySelector('parsererror')).toBeNull()

  return parsed
}

it('embeds the actual image bytes with a matching drawing relationship and content type', () => {
  const zip = parts()
  const media = Object.keys(zip).filter(name => name.startsWith('word/media/'))
  expect(media).toHaveLength(1)
  expect(zip[media[0]]).toEqual(Uint8Array.from(atob(image.split(',')[1]), c => c.charCodeAt(0)))
  const body = xml(strFromU8(zip['word/document.xml']))
  const blip = body.getElementsByTagNameNS('http://schemas.openxmlformats.org/drawingml/2006/main', 'blip')[0]
  const id = blip.getAttributeNS('http://schemas.openxmlformats.org/officeDocument/2006/relationships', 'embed')
  const rels = xml(strFromU8(zip['word/_rels/document.xml.rels']))
  const rel = [...rels.getElementsByTagName('Relationship')].find(node => node.getAttribute('Id') === id)
  expect(rel?.getAttribute('Target')).toBe(media[0].slice('word/'.length))
  expect(strFromU8(zip['[Content_Types].xml'])).toContain('image/png')
  expect(body.documentElement.textContent).toContain('Please review this diagram.')
})

it('uses readable Word styles and foldable detail headings that end before the next speaker', () => {
  const value: TranscriptDocument = {
    ...doc,
    options: { ...doc.options, include_tools: true, include_reasoning: true },
    messages: [
      {
        role: 'assistant',
        speaker: 'Alice',
        text: 'The answer.',
        tool_calls: [{ name: 'lookup', arguments: '{"q":"sample"}' }],
        reasoning: 'Stored explanation.'
      },
      { role: 'user', speaker: 'User', kind: 'aside', text: 'A side question.' }
    ]
  }

  const zip = parts(value)
  expect(zip['word/styles.xml']).toBeDefined()
  expect(zip['word/footer1.xml']).toBeDefined()
  const body = xml(strFromU8(zip['word/document.xml']))
  const ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
  const rows = [...body.getElementsByTagNameNS(ns, 'p')]
  const style = (p: Element) => p.getElementsByTagNameNS(ns, 'pStyle')[0]?.getAttributeNS(ns, 'val')
  const details = rows.filter(p => style(p) === 'Heading2')
  expect(details.map(p => p.textContent)).toEqual(['Reasoning', 'Reply', 'Tool: lookup'])

  for (const p of details.filter(p => p.textContent !== 'Reply')) {
    expect(p.getElementsByTagNameNS('http://schemas.microsoft.com/office/word/2012/wordml', 'collapsed')).toHaveLength(
      1
    )
  }

  const nextSpeaker = rows.find(p => p.textContent?.includes('Aside — /btw'))!
  expect(style(nextSpeaker)).toBe('Heading1')
  const styles = xml(strFromU8(zip['word/styles.xml']))
  expect(styles.getElementsByTagNameNS(ns, 'outlineLvl').length).toBeGreaterThanOrEqual(2)
  const expanded = parts({ ...value, options: { ...value.options, collapse_details: false } })
  expect(
    xml(strFromU8(expanded['word/document.xml'])).getElementsByTagNameNS(
      'http://schemas.microsoft.com/office/word/2012/wordml',
      'collapsed'
    )
  ).toHaveLength(0)
})
it('creates only explicitly enabled HTTP links and retains plain text when disabled', () => {
  const value: TranscriptDocument = {
    ...doc,
    messages: [
      {
        role: 'assistant',
        speaker: 'Alice',
        text: 'Read [the brief](https://example.com/brief?a=1&b=2). Never run javascript:alert(1).',
        attachments: [{ kind: 'file', name: 'Report', reference: 'https://example.com/report.pdf' }]
      }
    ]
  }

  const zip = parts(value)
  const body = xml(strFromU8(zip['word/document.xml']))
  const rels = xml(strFromU8(zip['word/_rels/document.xml.rels']))

  const links = [...rels.getElementsByTagName('Relationship')].filter(r =>
    r.getAttribute('Type')?.endsWith('/hyperlink')
  )

  expect(links.map(r => r.getAttribute('Target'))).toEqual([
    'https://example.com/brief?a=1&b=2',
    'https://example.com/report.pdf'
  ])
  expect(body.documentElement.textContent).toContain('the brief')
  expect(body.documentElement.textContent).toContain('javascript:alert(1)')
  const off = parts({ ...value, options: { ...value.options, include_links: false } })
  expect(
    xml(strFromU8(off['word/document.xml'])).getElementsByTagNameNS(
      'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'hyperlink'
    )
  ).toHaveLength(0)
  expect(xml(strFromU8(off['word/document.xml'])).documentElement.textContent).toContain(value.messages[0].text)
})
it('makes stored tool results foldable without hiding the next conversational turn', () => {
  const value: TranscriptDocument = {
    ...doc,
    options: { ...doc.options, include_tools: true },
    messages: [
      { role: 'assistant', speaker: 'Alice', text: 'Looking it up.' },
      { role: 'tool', speaker: 'search', text: 'A stored result.' },
      { role: 'assistant', speaker: 'Alice', text: 'Here is the answer.' }
    ]
  }

  const body = xml(strFromU8(parts(value)['word/document.xml']))
  const ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
  const rows = [...body.getElementsByTagNameNS(ns, 'p')]
  const result = rows.find(p => p.textContent?.includes('Tool result — search'))!
  expect(result).toBeDefined()
  expect(result.getElementsByTagNameNS(ns, 'pStyle')[0].getAttributeNS(ns, 'val')).toBe('Heading2')
  expect(
    result.getElementsByTagNameNS('http://schemas.microsoft.com/office/word/2012/wordml', 'collapsed')
  ).toHaveLength(1)
  expect(
    rows.filter(p => p.getElementsByTagNameNS(ns, 'pStyle')[0]?.getAttributeNS(ns, 'val') === 'Heading1')
  ).toHaveLength(2)
})

it('retains provenance when embedding an image', () => {
  const value: TranscriptDocument = {
    ...doc,
    messages: [
      {
        ...doc.messages[0],
        attachments: [
          {
            kind: 'image',
            name: 'img',
            data_url: image,
            reference: 'https://example.org/source.png',
            local: true,
            unavailable: 'source missing'
          }
        ]
      }
    ]
  }

  const zip = parts(value)
  expect(Object.keys(zip).some(name => name.startsWith('word/media/'))).toBe(true)
  const body = xml(strFromU8(zip['word/document.xml'])).documentElement.textContent
  expect(body).toContain('https://example.org/source.png')
  expect(body).toContain('local reference')
  expect(body).toContain('source missing')
})

it('links detail URLs without rewriting literal tool arguments', () => {
  const args = JSON.stringify({ text: '[label](https://example.org/tool)' })

  const value: TranscriptDocument = {
    ...doc,
    options: { ...doc.options, include_tools: true, include_reasoning: true },
    messages: [
      {
        role: 'assistant',
        speaker: 'Alice',
        text: 'Answer',
        reasoning: 'See https://example.org/reason',
        tool_calls: [{ name: 'run', arguments: args }]
      }
    ]
  }

  const zip = parts(value)
  const rels = xml(strFromU8(zip['word/_rels/document.xml.rels']))

  const targets = [...rels.getElementsByTagName('Relationship')]
    .filter(r => r.getAttribute('Type')?.endsWith('/hyperlink'))
    .map(r => r.getAttribute('Target'))

  expect(targets).toEqual(expect.arrayContaining(['https://example.org/tool', 'https://example.org/reason']))
  expect(xml(strFromU8(zip['word/document.xml'])).documentElement.textContent).toContain(args)
})

it('links embedded remote provenance only when enabled and keeps its text when disabled', () => {
  const value: TranscriptDocument = { ...doc, messages: [{ ...doc.messages[0], attachments: [{ kind: 'image', name: 'img', data_url: image, reference: 'https://example.org/source.png' }] }] }

  for (const enabled of [true, false]) {
    const zip = parts({ ...value, options: { ...value.options, include_links: enabled } })
    const body = xml(strFromU8(zip['word/document.xml']))
    const rels = xml(strFromU8(zip['word/_rels/document.xml.rels']))
    const links = [...rels.getElementsByTagName('Relationship')].filter(r => r.getAttribute('Type')?.endsWith('/hyperlink')).map(r => r.getAttribute('Target'))
    expect(body.documentElement.textContent).toContain('https://example.org/source.png')
    expect(links.includes('https://example.org/source.png')).toBe(enabled)
  }
})
