import { strFromU8, unzipSync } from 'fflate'
import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderTranscript } from './transcript-render'

const doc: TranscriptDocument = {
  version: 1,
  session_id: 's',
  title: 'Demo',
  exported_at: '2026-01-01T00:00:00Z',
  options: DEFAULT_TRANSCRIPT_OPTIONS,
  messages: [{ role: 'user', speaker: 'Jerry', text: 'hello & goodbye' }],
  warnings: []
}

it('produces an ODT with the mimetype stored first and a content manifest', () => {
  const result = renderTranscript(doc, 'odt')
  expect(result.encoding).toBe('base64')
  const bytes = Uint8Array.from(atob(result.content), c => c.charCodeAt(0))
  const view = new DataView(bytes.buffer)
  expect(view.getUint16(8, true)).toBe(0)
  const nameLength = view.getUint16(26, true)
  expect(strFromU8(bytes.subarray(30, 30 + nameLength))).toBe('mimetype')
  const entries = unzipSync(bytes)
  expect(strFromU8(entries.mimetype)).toBe('application/vnd.oasis.opendocument.text')
  expect(strFromU8(entries['META-INF/manifest.xml'])).toContain('content.xml')
  expect(strFromU8(entries['content.xml'])).toContain('hello &amp; goodbye')
})

it('produces a real DOCX package with a document and relationships', () => {
  const result = renderTranscript(doc, 'docx')
  expect(result.encoding).toBe('base64')
  const entries = unzipSync(Uint8Array.from(atob(result.content), c => c.charCodeAt(0)))
  expect(strFromU8(entries['[Content_Types].xml'])).toContain('wordprocessingml.document.main+xml')
  expect(strFromU8(entries['_rels/.rels'])).toContain('word/document.xml')
  expect(strFromU8(entries['word/document.xml'])).toContain('hello &amp; goodbye')
})
