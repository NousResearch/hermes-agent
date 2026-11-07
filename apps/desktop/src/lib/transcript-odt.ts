import { strToU8, zipSync } from 'fflate'

import { transcriptLayout } from './transcript-association'
import type { TranscriptDocument } from './transcript-document'
import { transcriptBase64, transcriptImage } from './transcript-media'
import { transcriptTextParts } from './transcript-text'

const mime = 'application/vnd.oasis.opendocument.text'

const x = (s: string) =>
  s
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '\ufffd')
    .replace(/[\s\S]/g, c => {
      const code = c.charCodeAt(0)

      return (code < 32 && code !== 9 && code !== 10 && code !== 13) || code === 0xfffe || code === 0xffff ? '' : c
    })
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')

const namespaces =
  'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" xmlns:xlink="http://www.w3.org/1999/xlink"'

function renderBody(doc: TranscriptDocument, media: Record<string, Uint8Array>, entries: string[]): string {
  const { associations, groups } = transcriptLayout(doc)
  const grouped = new Set(groups.flatMap(group => group.results))
  const activity = new Set(groups.filter(group => group.results.length).map(group => group.index))

  const inline = (line: string, links: boolean, literal: boolean) =>
    transcriptTextParts(line, links && doc.options.include_links, literal)
      .map(part => part.href
        ? `<text:a xlink:type="simple" xlink:href="${x(part.href)}">${x(part.text)}</text:a>`
        : x(part.text))
      .join('')

  const p = (s: string, links = true, style = 'Body', literal = false) =>
    `<text:p text:style-name="${style}">${s
      .split(/\r\n|\r|\n/)
      .map(line =>
        inline(line, links, literal)
          .replace(/ {2,}/g, spaces => `<text:s text:c="${spaces.length}"/>`)
          .replace(/\t/g, '<text:tab/>')
      )
      .join('<text:line-break/>')}</text:p>`

  const h = (s: string, n: number) =>
    `<text:h text:style-name="${n === 1 ? 'Title' : n === 2 ? 'Speaker' : 'Aux'}" text:outline-level="${n}">${x(s)}</text:h>`

  return [
    h(doc.title, 1),
    p(
      `Session: ${doc.session_id} · Exported: ${doc.exported_at}${doc.created_at ? ` · Created: ${doc.created_at}` : ''}${doc.model ? ` · Model: ${doc.model}` : ''}`,
      false,
      'Metadata'
    ),
    ...doc.warnings.map(w => p(`Warning: ${w}`, false, 'Warning')),
    ...doc.messages.flatMap((m, index) => [
      h(
        `${m.kind === 'notice' ? 'Notice' : m.role.toLowerCase() === 'tool' ? `Tool result — ${m.speaker}` : m.speaker}${m.timestamp ? ` · ${m.timestamp}` : ''}${m.kind === 'aside' ? ' · Aside — /btw' : ''}`,
        grouped.has(index) ? 4 : 2
      ),
      ...(associations[index] ? [p(associations[index]!, false, 'Metadata')] : []),
      ...(m.reasoning === undefined ? [] : [h('Reasoning', 3), p(m.reasoning, true, 'Reasoning')]),
      p(m.text, true, m.role.toLowerCase() === 'tool' ? 'Code' : 'Body', m.role.toLowerCase() === 'tool'),
      ...(activity.has(index) ? [h('Tool activity', 3)] : []),
      ...(m.tool_calls ?? []).flatMap((c, i) => [h(`Tool: ${c.name}${c.id || c.call_id || c.uid ? ` · Call ${i + 1}` : ''}`, activity.has(index) ? 4 : 3), p(c.arguments, true, 'Code', true)]),
      ...(m.attachments ?? []).map(a => {
        const img = a.kind === 'image' ? transcriptImage(a.data_url, 'odt') : undefined
        let drawing = ''

        if (img?.width && img.height) {
          const name = `Pictures/image${entries.length + 1}.${img.extension}`
          media[name] = img.bytes
          entries.push(`<manifest:file-entry manifest:full-path="${name}" manifest:media-type="${img.mime}"/>`)
          const scale = Math.min(12 / img.width, 10 / img.height, 0.0264583)
          const width = Math.max(0.1, img.width * scale).toFixed(3)
          const height = Math.max(0.1, img.height * scale).toFixed(3)
          drawing = `<text:p><draw:frame draw:style-name="ImageFrame" text:anchor-type="paragraph" svg:width="${width}cm" svg:height="${height}cm"><draw:image xlink:href="${name}" xlink:type="simple" xlink:show="embed" xlink:actuate="onLoad"/></draw:frame></text:p>`
        }

        return (
          p(
            `${a.kind}: ${a.name}${a.reference ? ` — ${a.reference}` : ''}${a.local ? ' (local reference; may be unavailable on another computer)' : ''}${a.unavailable ? ` — ${a.unavailable}` : a.kind === 'image' && !img ? ' — image unavailable or unsupported' : ''}`,
            !a.local,
            'Body',
            true
          ) + drawing
        )
      })
    ])
  ].join('')
}

export function renderOdt(doc: TranscriptDocument): string {
  const media: Record<string, Uint8Array> = {}
  const entries: string[] = []
  const body = renderBody(doc, media, entries)
  const content = `<?xml version="1.0" encoding="UTF-8"?><office:document-content ${namespaces} xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" office:version="1.2"><office:automatic-styles><style:style style:name="ImageFrame" style:family="graphic"/></office:automatic-styles><office:body><office:text>${body}</office:text></office:body></office:document-content>`
  const ns = `${namespaces} xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"`

  const style = (name: string, paragraph: string, text: string) =>
    `<style:style style:name="${name}" style:family="paragraph"><style:paragraph-properties ${paragraph}/><style:text-properties ${text}/></style:style>`

  const styles = `<?xml version="1.0" encoding="UTF-8"?><office:document-styles ${ns} office:version="1.2"><office:styles><style:default-style style:family="paragraph"><style:paragraph-properties fo:line-height="145%" fo:margin-bottom="0.18cm"/><style:text-properties fo:font-family="Liberation Sans" fo:font-size="11pt" fo:color="#253245"/></style:default-style>${style('Body', 'fo:line-height="145%" fo:margin-bottom="0.2cm"', 'fo:font-size="11pt"')}${style('Title', 'fo:margin-bottom="0.3cm"', 'fo:font-size="23pt" fo:font-weight="bold" fo:color="#172c49"')}${style('Speaker', 'fo:margin-top="0.55cm" fo:margin-bottom="0.12cm" fo:keep-with-next="always"', 'fo:font-size="13pt" fo:font-weight="bold" fo:color="#245682"')}${style('Aux', 'fo:margin-top="0.3cm" fo:keep-with-next="always"', 'fo:font-size="10pt" fo:font-weight="bold" fo:color="#536780"')}${style('Metadata', 'fo:margin-bottom="0.5cm"', 'fo:font-size="9pt" fo:color="#607087"')}${style('Reasoning', 'fo:margin-left="0.45cm" fo:margin-right="0.2cm" fo:margin-bottom="0.18cm"', 'fo:font-size="10pt" fo:color="#526078"')}${style('Code', 'fo:background-color="#f2f5f8" fo:padding="0.2cm" fo:margin-left="0.25cm"', 'fo:font-family="Liberation Mono" fo:font-size="9pt"')}${style('Warning', 'fo:background-color="#fff5dc" fo:padding="0.2cm"', 'fo:font-size="10pt" fo:color="#754b09"')}${style('HeaderFooter', '', 'fo:font-size="8pt" fo:color="#6d7b8c"')}</office:styles><office:automatic-styles><style:page-layout style:name="Page"><style:page-layout-properties fo:page-width="21cm" fo:page-height="29.7cm" style:print-orientation="portrait" fo:margin-top="2cm" fo:margin-bottom="2cm" fo:margin-left="2.2cm" fo:margin-right="2.2cm"/><style:header-style/><style:footer-style/></style:page-layout></office:automatic-styles><office:master-styles><style:master-page style:name="Standard" style:page-layout-name="Page"><style:header><text:p text:style-name="HeaderFooter">${x(doc.title)}</text:p></style:header><style:footer><text:p text:style-name="HeaderFooter">Transcript · Page <text:page-number text:select-page="current">1</text:page-number></text:p></style:footer></style:master-page></office:master-styles></office:document-styles>`
  const meta = `<?xml version="1.0" encoding="UTF-8"?><office:document-meta ${namespaces} xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" office:version="1.2"><office:meta><dc:title>${x(doc.title)}</dc:title><meta:generator>Hermes Agent</meta:generator></office:meta></office:document-meta>`
  const manifest = `<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2"><manifest:file-entry manifest:full-path="/" manifest:media-type="${mime}"/>${['content.xml', 'styles.xml', 'meta.xml'].map(name => `<manifest:file-entry manifest:full-path="${name}" manifest:media-type="text/xml"/>`).join('')}${entries.join('')}</manifest:manifest>`

  const bytes = zipSync({
    mimetype: [strToU8(mime), { level: 0 }],
    'content.xml': strToU8(content),
    'styles.xml': strToU8(styles),
    'meta.xml': strToU8(meta),
    'META-INF/manifest.xml': strToU8(manifest),
    ...media
  })

  return transcriptBase64(bytes)
}
