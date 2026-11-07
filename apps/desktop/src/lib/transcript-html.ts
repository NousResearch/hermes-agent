import { transcriptLayout } from './transcript-association'
import type { TranscriptAttachment, TranscriptDocument, TranscriptTurn } from './transcript-document'
import { escapeTranscriptXml as esc, transcriptBase64, transcriptImage } from './transcript-media'
import { safeTranscriptLink, transcriptTextParts } from './transcript-text'

const style = `
@page{size:A4;margin:18mm}
.technical-disclosure{border:0;padding:0;margin:.35rem 0 0 .7rem;font-size:.85rem}.technical-disclosure>summary{line-height:1.5;min-height:30px;padding:.3rem .15rem;color:var(--muted);font-weight:600;overflow-wrap:anywhere}.technical-disclosure>.body,.print-detail{font-size:.85rem;line-height:1.55;color:var(--muted);margin:.15rem 0 .45rem .7rem;padding:.35rem .7rem;border-left:2px solid var(--line)}.turn.tool{background:transparent;border:0;border-left:1px solid var(--line);border-radius:0;padding:.15rem 0 .15rem .7rem;margin:.3rem 0 .5rem 1.25rem}.tool .byline h2,.tool .byline time{font-size:.8rem;color:var(--muted)}@media(max-width:480px){.technical-disclosure>summary{min-height:44px;padding:.65rem .15rem}.turn.tool{margin-left:.75rem}}@media print{.print-detail{color:#333}.turn.tool{border:0;border-left:1px solid #bbb;margin-left:10pt}}
:root{color-scheme:light dark;--bg:#f3f5f9;--card:#fff;--ink:#172033;--muted:#526078;--line:#d7deea;--accent:#3157a4;--aside:#f4f1ff;--notice:#fff7e6}
@media(prefers-color-scheme:dark){:root{--bg:#141923;--card:#202938;--ink:#edf1f8;--muted:#b4c1d3;--line:#455266;--accent:#a9c3ff;--aside:#302a48;--notice:#413725}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:58rem;margin:auto;padding:2.5rem 1rem 5rem}header{margin-bottom:2rem}h1{font-size:clamp(1.6rem,4vw,2.3rem);line-height:1.2;overflow-wrap:anywhere}h2{font-size:1rem;margin:0}p{margin:.5rem 0}a{color:var(--accent)}a:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:3px}.meta,time{color:var(--muted);font-size:.86rem;overflow-wrap:anywhere}.turn{background:var(--card);border:1px solid var(--line);border-radius:.65rem;padding:1rem 1.25rem;margin:.8rem 0}.aside{background:var(--aside);border-left:4px solid var(--accent)}.notice{background:var(--notice)}.tool{border-left:4px solid var(--muted)}.byline{display:flex;gap:.5rem 1rem;align-items:baseline;flex-wrap:wrap}.body{white-space:pre-wrap;overflow-wrap:anywhere;margin:.65rem 0;font:inherit}.attachment{overflow-wrap:anywhere;font-size:.9rem;color:var(--muted)}img{display:block;max-width:100%;height:auto;max-height:38rem;object-fit:contain;margin:.7rem 0}details{border-top:1px solid var(--line);padding:.6rem 0}summary{cursor:pointer;font-weight:600}.print-detail{display:none}.warning{border-left:3px solid #b77b20;padding-left:.7rem}@media(max-width:480px){main{padding:1.2rem .65rem 3rem}.turn{padding:.8rem}}@media print{body{background:white;color:black}main{max-width:none;padding:0}.turn{border:1px solid #bbb;break-inside:avoid;background:white}.print-detail{display:block}details{display:none}a{color:black}.meta,time,.attachment{color:#333}}
`

function text(value: string, links: boolean, literal = false): string {
  return transcriptTextParts(value, links, literal)
    .map(p => (p.href ? `<a href="${esc(p.href)}" rel="noreferrer noopener">${esc(p.text)}</a>` : esc(p.text)))
    .join('')
}

function disclosure(label: string, value: string, collapse: boolean, links: boolean, literal = false): string {
  const body = `<div class="body">${text(value, links, literal)}</div>`

  return `<details class="technical-disclosure${label === 'Reasoning' ? ' reasoning-row' : ''}"${collapse ? '' : ' open'}><summary>${esc(label)}</summary>${body}</details><div class="print-detail"><strong>${esc(label)}</strong>${body}</div>`
}

function attachment(a: TranscriptAttachment, links: boolean): string {
  const image = a.kind === 'image' ? transcriptImage(a.data_url) : undefined

  const picture = image
    ? `<img alt="${esc(a.name)}" src="data:${image.mime};base64,${transcriptBase64(image.bytes)}">`
    : ''

  const ref = a.reference
    ? ` — ${links && !a.local && safeTranscriptLink(a.reference) ? `<a href="${esc(a.reference)}" rel="noreferrer noopener">${esc(a.reference)}</a>` : esc(a.reference)}`
    : ''

  const notes = `${a.local ? ' (local reference; may be unavailable on another computer)' : ''}${a.unavailable ? ` — ${a.unavailable}` : ''}${a.kind === 'image' && !image && !a.unavailable ? ' — image not embedded' : ''}`

  return `<div class="attachment">${esc(a.kind)}: ${esc(a.name)}${ref}${esc(notes)}${picture}</div>`
}

function turn(m: TranscriptTurn, doc: TranscriptDocument, association?: string, nested = ''): string {
  const links = doc.options.include_links
  const collapse = doc.options.collapse_details
  const tool = m.role.toLowerCase() === 'tool'

  const label =
    m.kind === 'aside' ? ' · Aside — /btw' : m.kind === 'notice' ? ' · Notice' : tool ? ' · Tool result' : ''

  const body = tool
    ? disclosure('Tool result', m.text, collapse, links, true)
    : `<div class="body">${text(m.text, links)}</div>`

  return `<article class="turn ${m.kind === 'aside' ? 'aside' : m.kind === 'notice' ? 'notice' : tool ? 'tool' : 'message'}"><div class="byline"><h2>${esc(m.speaker)}${label}</h2>${m.timestamp ? `<time>${esc(m.timestamp)}</time>` : ''}</div>${association ? `<div class="meta">${esc(association)}</div>` : ''}${m.reasoning === undefined ? '' : disclosure('Reasoning', m.reasoning, collapse, links)}${body}${(m.tool_calls ?? []).map((c, i) => disclosure(`Tool: ${c.name}${c.id || c.call_id || c.uid ? ` · Call ${i + 1}` : ''}`, c.arguments, collapse, links, true)).join('')}${(m.attachments ?? []).map(a => attachment(a, links)).join('')}${nested}</article>`
}

export function renderHtml(doc: TranscriptDocument): string {
  const metadata = `Session: ${esc(doc.session_id)} · Exported: ${esc(doc.exported_at)}${doc.model ? ` · Model: ${esc(doc.model)}` : ''}`
  const { associations, groups } = transcriptLayout(doc)

  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>${esc(doc.title)}</title><style>${style}</style></head><body><main><header><h1>${esc(doc.title)}</h1><p class="meta">${metadata}</p></header>${doc.warnings.map(w => `<p class="warning">Warning: ${esc(w)}</p>`).join('')}${groups.map(({ index, results }) => turn(doc.messages[index], doc, associations[index], results.map(i => turn(doc.messages[i], doc, associations[i])).join(''))).join('')}</main></body></html>`
}
