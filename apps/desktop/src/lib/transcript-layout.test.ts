import { strFromU8, unzipSync } from 'fflate'
import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderDocx } from './transcript-docx'
import { renderHtml } from './transcript-html'
import { renderMarkdown } from './transcript-markdown'
import { renderOdt } from './transcript-odt'

const doc: TranscriptDocument = { version: 1, session_id: 's', title: 'Layout', exported_at: 'now', warnings: [], options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_reasoning: true, include_tools: true }, messages: [{ role: 'assistant', speaker: 'Hermes', text: 'REPLY-unique', reasoning: 'REASON-unique', tool_calls: [{ name: 'run', arguments: '{"x":"literal"}' }] }, { role: 'tool', speaker: 'run', timestamp: 'later', text: 'RESULT-unique' }, { role: 'user', speaker: 'User', text: 'NEXT-unique' }] }
const part = (base64: string, name: string) => strFromU8(unzipSync(Uint8Array.from(atob(base64), c => c.charCodeAt(0)))[name])

it('places reasoning before the reply in every format and keeps tool results in chronological position', () => {
  const outputs = [renderHtml(doc), renderMarkdown(doc), part(renderDocx(doc), 'word/document.xml'), part(renderOdt(doc), 'content.xml')]

  for (const output of outputs) {
    expect(output.indexOf('REASON-unique')).toBeLessThan(output.indexOf('REPLY-unique'))
    expect(output.indexOf('REPLY-unique')).toBeLessThan(output.indexOf('RESULT-unique'))
    expect(output.indexOf('RESULT-unique')).toBeLessThan(output.indexOf('NEXT-unique'))
    expect(output).toContain('literal')
  }
})

it('ends the collapsed Word reasoning section at a separate Reply heading', () => {
  const xml = new DOMParser().parseFromString(part(renderDocx(doc), 'word/document.xml'), 'application/xml')
  const ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
  const rows = [...xml.getElementsByTagNameNS(ns, 'p')]
  const reasoning = rows.findIndex(p => p.textContent === 'Reasoning')
  const reply = rows.findIndex(p => p.textContent === 'Reply')
  const actual = rows.findIndex(p => p.textContent === 'REPLY-unique')
  expect(reasoning).toBeGreaterThan(-1)
  expect(reply).toBeGreaterThan(reasoning)
  expect(actual).toBeGreaterThan(reply)
  expect(rows[reply].getElementsByTagNameNS(ns, 'pStyle')[0]?.getAttributeNS(ns, 'val')).toBe('Heading2')
  expect(rows[reply].getElementsByTagNameNS('http://schemas.microsoft.com/office/word/2012/wordml', 'collapsed')).toHaveLength(0)
})

it('keeps Markdown details visible inline even when collapse is requested', () => {
  const output = renderMarkdown(doc)
  expect(output).not.toContain('<details')
  expect(output).toContain('REASON-unique')
  expect(output).toContain('RESULT-unique')
})
