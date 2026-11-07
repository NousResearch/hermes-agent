import { writeFileSync } from 'node:fs'

import { strFromU8, unzipSync } from 'fflate'
import { describe, expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderOdt } from './transcript-odt'

const png = 'iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAIAAAA8r+mnAAAAEklEQVR4nGP8z4AdMOEQJ0MCAGSRAQfIidsoAAAAAElFTkSuQmCC'

const doc = (): TranscriptDocument => ({
  version: 1,
  session_id: 'session',
  title: 'Example',
  exported_at: '2026-01-01T00:00:00Z',
  options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_tools: true, include_reasoning: true },
  warnings: [],
  messages: [
    {
      role: 'user',
      speaker: 'Alice',
      text: 'Hello',
      attachments: [{ kind: 'image', name: 'pixel', data_url: `data:image/png;base64,${png}` }]
    }
  ]
})

const unpack = (d: TranscriptDocument) => unzipSync(Uint8Array.from(atob(renderOdt(d)), c => c.charCodeAt(0)))

const xml = (data: Uint8Array) => {
  const result = new DOMParser().parseFromString(strFromU8(data), 'application/xml')
  expect(result.querySelector('parsererror')).toBeNull()

  return result
}

describe('ODT export', () => {
  it('labels stored tool results and notices in order with only stored timestamps', () => {
    const d = doc()
    d.messages = [
      { role: 'tool', speaker: 'shell', timestamp: '2026-01-02T03:04:05Z', text: 'first result' },
      { role: 'assistant', speaker: 'Hermes', kind: 'notice', text: 'second notice' }
    ]
    const content = xml(unpack(d)['content.xml'])
    const text = content.documentElement.textContent ?? ''

    const headings = Array.from(
      content.getElementsByTagNameNS('urn:oasis:names:tc:opendocument:xmlns:text:1.0', 'h')
    ).map(e => e.textContent)

    expect(headings).toEqual(['Example', 'Tool result — shell · 2026-01-02T03:04:05Z', 'Notice'])
    expect(text.indexOf('first result')).toBeLessThan(text.indexOf('second notice'))
  })
  it('toggles hyperlinks while retaining remote and local reference text', () => {
    const d = doc()
    d.messages[0].text = 'Visit https://example.org/a?x=1&y=2 now'
    d.messages[0].attachments = [
      { kind: 'file', name: 'remote', reference: 'https://example.org/file' },
      { kind: 'file', name: 'local', local: true, reference: 'https://example.org/private' }
    ]
    const ns = 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'
    let content = xml(unpack(d)['content.xml'])
    let links = Array.from(content.getElementsByTagNameNS(ns, 'a'))
    expect(links.map(a => a.getAttributeNS('http://www.w3.org/1999/xlink', 'href'))).toEqual([
      'https://example.org/a?x=1&y=2',
      'https://example.org/file'
    ])
    expect(content.documentElement.textContent).toContain('local reference; may be unavailable on another computer')
    d.options.include_links = false
    content = xml(unpack(d)['content.xml'])
    links = Array.from(content.getElementsByTagNameNS(ns, 'a'))
    expect(links).toHaveLength(0)
    expect(content.documentElement.textContent).toContain('https://example.org/file')
  })
  it('creates styled ODF 1.2 package with manifest, metadata and page layout', () => {
    const d = doc()
    d.warnings = ['warning']
    d.messages[0].reasoning = 'Thinking'
    d.messages[0].tool_calls = [{ name: 'run', arguments: 'one\n  two' }]
    const binary = Uint8Array.from(atob(renderOdt(d)), c => c.charCodeAt(0))
    const view = new DataView(binary.buffer)

    if (process.env.ODT_SAMPLE_PATH) {
      writeFileSync(process.env.ODT_SAMPLE_PATH, binary)
    }

    expect(view.getUint16(8, true)).toBe(0)
    expect(new TextDecoder().decode(binary.subarray(30, 38))).toBe('mimetype')
    const parts = unzipSync(binary)

    for (const name of ['content.xml', 'styles.xml', 'meta.xml', 'META-INF/manifest.xml']) {
      expect(xml(parts[name]).documentElement).toBeTruthy()
    }

    const manifest = xml(parts['META-INF/manifest.xml'])
    const mn = 'urn:oasis:names:tc:opendocument:xmlns:manifest:1.0'

    const paths = Array.from(manifest.getElementsByTagNameNS(mn, 'file-entry')).map(e =>
      e.getAttributeNS(mn, 'full-path')
    )

    expect(paths).toEqual(expect.arrayContaining(['/', 'content.xml', 'styles.xml', 'meta.xml', 'Pictures/image1.png']))
    const styles = xml(parts['styles.xml'])
    const sn = 'urn:oasis:names:tc:opendocument:xmlns:style:1.0'
    expect(Array.from(styles.getElementsByTagNameNS(sn, 'style')).map(e => e.getAttributeNS(sn, 'name'))).toEqual(
      expect.arrayContaining(['Title', 'Speaker', 'Aux', 'Code', 'Warning'])
    )
    expect(styles.getElementsByTagNameNS(sn, 'page-layout').length).toBeGreaterThan(0)
    expect(styles.getElementsByTagNameNS(sn, 'header').length).toBeGreaterThan(0)
    expect(styles.getElementsByTagNameNS(sn, 'footer').length).toBeGreaterThan(0)
    const content = xml(parts['content.xml'])
    expect(
      content.getElementsByTagNameNS('urn:oasis:names:tc:opendocument:xmlns:office:1.0', 'automatic-styles').length
    ).toBe(1)
    expect(
      Array.from(content.getElementsByTagNameNS('urn:oasis:names:tc:opendocument:xmlns:text:1.0', 'h')).map(
        e => e.textContent
      )
    ).toEqual(['Example', 'Alice', 'Reasoning', 'Tool: run'])
  })
  it('rejects malformed untrusted media and unsafe links while keeping XML valid', () => {
    const d = doc()
    d.title = 'Bad \u0000 \ud800 title'
    d.messages[0].text = 'javascript:https://evil.test file:///local'
    d.messages[0].attachments = [
      { kind: 'image', name: 'forged', data_url: 'data:image/jpeg;base64,/9j/2Q==' },
      { kind: 'image', name: 'svg', data_url: 'data:image/svg+xml;base64,PHN2Zy8+' }
    ]
    const parts = unpack(d)
    expect(Object.keys(parts).filter(k => k.startsWith('Pictures/'))).toEqual([])
    const content = xml(parts['content.xml'])
    xml(parts['styles.xml'])
    xml(parts['meta.xml'])
    expect(content.getElementsByTagNameNS('urn:oasis:names:tc:opendocument:xmlns:text:1.0', 'a').length).toBe(0)
    expect(content.documentElement.textContent).toContain('image unavailable or unsupported')
  })
  it('embeds GIF and WebP data with declared MIME types', () => {
    const d = doc()
    d.messages[0].attachments = [
      {
        kind: 'image',
        name: 'gif',
        data_url: 'data:image/gif;base64,R0lGODdhCAAEAIEAAP8AAAAAAAAAAAAAACwAAAAACAAEAAAIDAABCBxIsKDBgwQDAgA7'
      },
      {
        kind: 'image',
        name: 'webp',
        data_url:
          'data:image/webp;base64,UklGRjwAAABXRUJQVlA4IDAAAADQAQCdASoIAAQAAUAmJaACdLoB+AADsAD+8ut//NgVzXPv9//S4P0uD9Lg/9KQAAA='
      }
    ]
    const parts = unpack(d)
    expect(parts['Pictures/image1.gif']).toBeDefined()
    expect(parts['Pictures/image2.webp']).toBeDefined()
    const manifest = xml(parts['META-INF/manifest.xml'])
    expect(manifest.documentElement.textContent).not.toBeNull()
    const ns = 'urn:oasis:names:tc:opendocument:xmlns:manifest:1.0'

    const types = Array.from(manifest.getElementsByTagNameNS(ns, 'file-entry')).map(e =>
      e.getAttributeNS(ns, 'media-type')
    )

    expect(types).toEqual(expect.arrayContaining(['image/gif', 'image/webp']))
  })
  it('embeds valid image bytes with a package relationship', () => {
    const parts = unpack(doc())
    expect(parts['Pictures/image1.png']).toEqual(Uint8Array.from(atob(png), c => c.charCodeAt(0)))
    const content = xml(parts['content.xml'])
    const image = content.getElementsByTagNameNS('urn:oasis:names:tc:opendocument:xmlns:drawing:1.0', 'image')[0]
    expect(image.getAttributeNS('http://www.w3.org/1999/xlink', 'href')).toBe('Pictures/image1.png')
    expect(strFromU8(parts['META-INF/manifest.xml'])).toContain('image/png')
  })
})
