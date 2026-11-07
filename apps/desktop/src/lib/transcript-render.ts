import { parseTranscriptDocument, type TranscriptDocument } from './transcript-document'
import { renderDocx } from './transcript-docx'
import { renderHtml } from './transcript-html'
import { renderMarkdown } from './transcript-markdown'
import { renderOdt } from './transcript-odt'

export function renderTranscript(
  document: TranscriptDocument,
  format: 'docx' | 'odt' | 'json' | 'md' | 'html'
): { content: string; encoding?: 'base64' } {
  const doc = parseTranscriptDocument(JSON.stringify(document))

  if (format === 'odt') {return { content: renderOdt(doc), encoding: 'base64' }}

  if (format === 'docx') {return { content: renderDocx(doc), encoding: 'base64' }}

  if (format === 'json') {return { content: JSON.stringify(doc, null, 2) }}

  if (format === 'md') {return { content: renderMarkdown(doc) }}

  if (format !== 'html') {throw new Error(`Unsupported transcript format: ${format}`)}

  return { content: renderHtml(doc) }
}
