import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderTranscript } from './transcript-render'

const doc: TranscriptDocument = {
  version: 1,
  session_id: 's',
  title: 'Demo',
  exported_at: '2026-01-01T00:00:00Z',
  options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_reasoning: true },
  messages: [
    { role: 'assistant', speaker: 'A < B', text: 'Hello <script>alert(1)</script>', reasoning: 'stored thought' }
  ],
  warnings: []
}

it('shows attachment references without loading them remotely', () => {
  const withFile: TranscriptDocument = {
    ...doc,
    messages: [
      {
        role: 'user',
        speaker: 'U',
        text: '',
        attachments: [{ kind: 'file', name: 'notes.txt', reference: 'C:/private/notes.txt', local: true }]
      }
    ]
  }

  const html = renderTranscript(withFile, 'html').content
  expect(html).toContain('notes.txt')
  expect(html).toContain('may be unavailable on another computer')
  expect(html).not.toContain('src="C:/')
})

it('renders Markdown with readable content and a native reasoning disclosure', () => {
  const markdown = renderTranscript(doc, 'md').content
  expect(markdown).toContain('Hello &lt;script&gt;')
  expect(markdown).toContain('**Reasoning**')
  expect(markdown).toContain('stored thought')
})

it('renders structured JSON without dropping transcript data', () => {
  expect(JSON.parse(renderTranscript(doc, 'json').content)).toEqual(doc)
})

it('renders escaped HTML with a closed native reasoning disclosure', () => {
  const html = renderTranscript(doc, 'html').content
  expect(html).toContain('&lt;script&gt;')
  expect(html).not.toContain('<script>')
  expect(html).toContain('<summary>Reasoning</summary>')
  expect(html).toContain('stored thought')
})
