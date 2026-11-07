import { transcriptLayout } from './transcript-association'
import type { TranscriptAttachment, TranscriptDocument, TranscriptTurn } from './transcript-document'
import { markdownProse } from './transcript-markdown-prose'
import { escapeTranscriptXml as esc, transcriptBase64, transcriptImage } from './transcript-media'
import { safeTranscriptLink, transcriptTextParts } from './transcript-text'

// Untrusted delimiters must not consume generated links or create destinations.
const literal = (s: string): string => esc(s)
  .replace(/([[\]\\!`])/g, '\\$1')
  .replace(/~(?=~~)/g, '\\~')

const plain = (s: string): string => {
  let length = 1

  for (const match of s.matchAll(/`+/g)) {
    length = Math.max(length, match[0].length + 1)
  }

  const ticks = '`'.repeat(length)

  return `${ticks} ${s.replace(/[\r\n]+/g, ' ')} ${ticks}`
}

function text(value: string, links: boolean, literalText = false): string {
  if (literalText) {
    const body = transcriptTextParts(value, links, true)
      .map(p => p.href ? `<a href="${esc(p.href)}" rel="noreferrer noopener">${esc(p.text)}</a>` : esc(p.text))
      .join('')

    return `<pre>${body}</pre>`
  }

  if (!links) {
    return `<pre>${esc(value)}</pre>`
  }

  if (/^\s*[[{]/.test(value)) {
    try {
      JSON.parse(value)

      return text(value, links, true)
    } catch {
      // A Markdown paragraph may begin with a bracket without being JSON.
    }
  }

  return markdownProse(value)
}

function disclosure(label: string, value: string, links: boolean, literalText = false): string {
  return `**${literal(label)}**\n\n${text(value, links, literalText)}`
}

function attachment(a: TranscriptAttachment, links: boolean): string {
  const image = a.kind === 'image' ? transcriptImage(a.data_url) : undefined

  const ref = a.reference
    ? ` — ${links && !a.local && safeTranscriptLink(a.reference) ? `[${literal(a.reference)}](<${esc(a.reference)}>)` : plain(a.reference)}`
    : ''

  const notes = `${a.local ? ' (local reference; may be unavailable on another computer)' : ''}${a.unavailable ? ` — ${a.unavailable}` : ''}${a.kind === 'image' && !image && !a.unavailable ? ' — image not embedded' : ''}`

  return `${literal(a.kind)}: ${plain(a.name)}${ref}${notes ? ` ${plain(notes)}` : ''}${image ? `\n\n![${literal(a.name)}](data:${image.mime};base64,${transcriptBase64(image.bytes)})` : ''}`
}

function turn(m: TranscriptTurn, doc: TranscriptDocument, association?: string): string {
  const links = doc.options.include_links

  const tool = m.role.toLowerCase() === 'tool'

  const label =
    m.kind === 'aside' ? ' · Aside — /btw' : m.kind === 'notice' ? ' · Notice' : tool ? ' · Tool result' : ''

  const body = tool ? disclosure('Tool result', m.text, links, true) : text(m.text, links)

  return [
    `## ${plain(m.speaker)}${label}`,
    association || '',
    m.timestamp ? plain(m.timestamp) : '',
    m.reasoning === undefined ? '' : disclosure('Reasoning', m.reasoning, links).split('\n').map(line => `> ${line}`).join('\n'),
    body,
    ...(m.tool_calls ?? []).map((c, i) => disclosure(`Tool: ${c.name}${c.id || c.call_id || c.uid ? ` · Call ${i + 1}` : ''}`, c.arguments, links, true)),
    ...(m.attachments ?? []).map(a => attachment(a, links))
  ]
    .filter(Boolean)
    .join('\n\n')
}

export function renderMarkdown(doc: TranscriptDocument): string {
  const { associations, groups } = transcriptLayout(doc)

  return (
    [
      `# ${plain(doc.title)}`,
      `Session: ${plain(doc.session_id)} · Exported: ${plain(doc.exported_at)}${doc.model ? ` · Model: ${plain(doc.model)}` : ''}`,
      'Technical details are shown inline; inline image support varies by Markdown viewer.',
      ...doc.warnings.map(w => `Warning: ${plain(w)}`),
      ...groups.map(({ index, results }) => [
        turn(doc.messages[index], doc, associations[index]),
        ...results.map(i => turn(doc.messages[i], doc, associations[i]).replace(/^## /, '**').replace(/\n/, '**\n').split('\n').map(line => `> ${line}`).join('\n'))
      ].join('\n\n'))
    ].join('\n\n') + '\n'
  )
}
