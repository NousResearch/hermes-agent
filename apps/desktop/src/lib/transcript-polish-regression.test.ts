import { strFromU8, unzipSync } from 'fflate'
import remarkGfm from 'remark-gfm'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderDocx } from './transcript-docx'
import { renderHtml } from './transcript-html'
import { renderMarkdown } from './transcript-markdown'
import { renderOdt } from './transcript-odt'

const doc: TranscriptDocument = { version: 1, session_id: 's', title: 't', exported_at: 'now', warnings: [], options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_links: true, include_reasoning: true }, messages: [{ role: 'assistant', speaker: 'Hermes', text: 'REPLY', reasoning: 'thought [safe](https://example.org)\n\n```js\nconst x = 1\n```' }] }

it('nests an adjacent identified result under its assistant in all readers', () => {
  const value: TranscriptDocument = { ...doc, messages: [
    { role: 'assistant', speaker: 'Hermes', text: 'REPLY', tool_calls: [{ name: 'run', arguments: 'ARGS', id: 'x' }] },
    { role: 'tool', speaker: 'Runner', text: 'RESULT', timestamp: 'later', tool_call_id: 'x' },
    { role: 'user', speaker: 'User', text: 'NEXT' }
  ] }

  const html = new DOMParser().parseFromString(renderHtml(value), 'text/html')
  expect(html.querySelector('article.message article.tool')?.textContent).toContain('RESULT')
  const md = unified().use(remarkParse).use(remarkGfm).parse(renderMarkdown(value))
  expect(md.children.some(node => node.type === 'blockquote' && JSON.stringify(node).includes('RESULT'))).toBe(true)
  const part = (base64: string, name: string) => strFromU8(unzipSync(Uint8Array.from(atob(base64), c => c.charCodeAt(0)))[name])
  const word = new DOMParser().parseFromString(part(renderDocx(value), 'word/document.xml'), 'application/xml')
  const w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
  const heading = [...word.getElementsByTagNameNS(w, 'p')].find(p => p.textContent?.includes('Tool result — Runner'))
  expect(heading?.getElementsByTagNameNS(w, 'pStyle')[0]?.getAttributeNS(w, 'val')).toBe('Heading3')
  const odt = new DOMParser().parseFromString(part(renderOdt(value), 'content.xml'), 'application/xml')
  const t = 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'
  const oh = [...odt.getElementsByTagNameNS(t, 'h')].find(h => h.textContent?.includes('Tool result — Runner'))
  expect(oh?.getAttributeNS(t, 'outline-level')).toBe('4')
})

it('keeps ambiguous or interleaved results at their original positions', () => {
  const value: TranscriptDocument = { ...doc, messages: [
    { role: 'assistant', speaker: 'Hermes', text: 'CALL', tool_calls: [{ name: 'run', arguments: 'ARG', id: 'x' }] },
    { role: 'tool', speaker: 'Runner', text: 'UNKNOWN' },
    { role: 'tool', speaker: 'Runner', text: 'LATE', tool_call_id: 'x' }
  ] }

  const page = new DOMParser().parseFromString(renderHtml(value), 'text/html')
  expect(page.querySelector('article.message article.tool')).toBeNull()
  const rows = [...page.querySelectorAll('main > article')]
  expect(rows).toHaveLength(3)
  expect(rows[1].textContent).toContain('Unassociated tool result (original position)')
  expect(rows[2].textContent).toContain('Result for assistant turn 1, call 1')
})

it.each([true, false])('keeps included activity present with collapse=%s', collapse_details => {
  const value: TranscriptDocument = { ...doc, options: { ...doc.options, include_tools: true, collapse_details }, messages: [
    { role: 'assistant', speaker: 'Hermes', text: 'REPLY', tool_calls: [{ name: 'run', arguments: 'ARG', id: 'x' }] },
    { role: 'tool', speaker: 'Runner', text: 'RESULT', tool_call_id: 'x' }
  ] }

  const page = new DOMParser().parseFromString(renderHtml(value), 'text/html')
  const details = [...page.querySelectorAll('details')]
  expect(details).toHaveLength(2)
  expect(details.every(item => item.open === !collapse_details)).toBe(true)
  expect(page.querySelector('article.message article.tool')?.textContent).toContain('RESULT')
})

it('prints on A4 with explicit safe margins', () => {
  const page = new DOMParser().parseFromString(renderHtml(doc), 'text/html')
  const css = page.querySelector('style')!.textContent!
  expect(css).toMatch(/@page\s*\{\s*size:\s*A4;\s*margin:\s*18mm\s*;?\s*\}/)
})

it('places reasoning including links and code in a blockquote before the reply', () => {
  const tree = unified().use(remarkParse).use(remarkGfm).parse(renderMarkdown(doc))
  const quote = tree.children.find(node => node.type === 'blockquote')
  expect(quote).toBeDefined()
  expect(JSON.stringify(quote)).toContain('https://example.org')
  expect(JSON.stringify(quote)).toContain('const x = 1')
  expect(JSON.stringify(quote)).not.toContain('REPLY')
  expect(tree.children.indexOf(quote!)).toBeLessThan(tree.children.findIndex(node => JSON.stringify(node).includes('REPLY')))
})
