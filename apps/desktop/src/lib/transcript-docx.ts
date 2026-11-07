import { strToU8, zipSync } from 'fflate'

import { transcriptLayout } from './transcript-association'
import type { TranscriptAttachment, TranscriptDocument, TranscriptTurn } from './transcript-document'
import { DOCX_FOOTER, DOCX_SETTINGS, DOCX_STYLES } from './transcript-docx-styles'
import { transcriptBase64, transcriptImage, escapeTranscriptXml as x } from './transcript-media'
import { transcriptTextParts } from './transcript-text'

const W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
const R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
const PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'
const XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'

function wordRun(text: string, properties = ''): string {
  const content = text
    .replace(/\r\n?/g, '\n')
    .split(/(\n|\t)/)
    .map(part => {
      if (part === '\n') {
        return '<w:br/>'
      }

      if (part === '\t') {
        return '<w:tab/>'
      }

      return `<w:t xml:space="preserve">${x(part)}</w:t>`
    })
    .join('')

  return `<w:r>${properties ? `<w:rPr>${properties}</w:rPr>` : ''}${content}</w:r>`
}

function paragraph(text: string, properties = '', runProperties = '', state?: PackageState, literal = false): string {
  const content = transcriptTextParts(text, state?.links === true, literal)
    .map(part => {
      if (!part.href || !state) {
        return wordRun(part.text, runProperties)
      }

      const id = `link${state.nextId++}`
      state.relationships.push(
        `<Relationship Id="${id}" Type="${R}/hyperlink" Target="${x(part.href)}" TargetMode="External"/>`
      )

      return `<w:hyperlink r:id="${id}">${wordRun(part.text, '<w:rStyle w:val="Hyperlink"/>')}</w:hyperlink>`
    })
    .join('')

  return `<w:p>${properties ? `<w:pPr>${properties}</w:pPr>` : ''}${content}</w:p>`
}

interface PackageState {
  files: Record<string, Uint8Array>
  relationships: string[]
  imageTypes: Map<string, string>
  nextId: number
  links: boolean
}

function attachment(a: TranscriptAttachment, state: PackageState): string {
  const image = a.kind === 'image' ? transcriptImage(a.data_url) : undefined

  if (image?.width && image.height && image.extension !== 'webp') {
    const id = state.nextId++
    const name = `image${id}.${image.extension}`
    state.files[`word/media/${name}`] = image.bytes
    state.imageTypes.set(image.extension, image.mime)
    state.relationships.push(`<Relationship Id="image${id}" Type="${R}/image" Target="media/${name}"/>`)
    const scale = Math.min((6.5 * 914400) / image.width, (6 * 914400) / image.height, 9525)
    const cx = Math.round(image.width * scale)
    const cy = Math.round(image.height * scale)

    return `<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="${cx}" cy="${cy}"/><wp:docPr id="${id}" name="${x(a.name)}" descr="${x(a.name)}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="${id}" name="${x(a.name)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="image${id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="${cx}" cy="${cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>${paragraph(`${a.kind}: ${a.name}${a.reference ? ` — ${a.reference}` : ''}${a.local ? ' (local reference; may be unavailable on another computer)' : ''}${a.unavailable ? ` — ${a.unavailable}` : ''}`, '', '', a.local ? undefined : state)}`
  }

  const note = a.unavailable || (a.data_url ? 'This image format cannot be embedded in this Word document.' : '')

  return paragraph(
    `${a.kind}: ${a.name}${a.reference ? ` — ${a.reference}` : ''}${a.local ? ' (local reference; may be unavailable on another computer)' : ''}${note ? ` — ${note}` : ''}`,
    '',
    '',
    a.local ? undefined : state
  )
}

function turn(message: TranscriptTurn, state: PackageState, collapse: boolean, association?: string, grouped = false, activity = false): string {
  const heading = `<w:pStyle w:val="Heading2"/>${collapse ? '<w15:collapsed/>' : ''}`
  const activityHeading = `<w:pStyle w:val="Heading3"/>${collapse ? '<w15:collapsed/>' : ''}`
  const caption = `${message.speaker} · ${message.timestamp || 'Time not recorded'}${message.kind === 'aside' ? ' · Aside — /btw' : ''}${message.kind === 'notice' ? ' · Notice' : ''}`
  const toolResult = message.role.toLowerCase() === 'tool'

  return [
    paragraph(toolResult ? `Tool result — ${caption}${association ? ` · ${association}` : ''}` : caption, toolResult ? (grouped ? activityHeading : heading) : '<w:pStyle w:val="Heading1"/>'),
    ...(!toolResult && association ? [paragraph(association, '<w:pStyle w:val="Metadata"/>')] : []),
    ...(message.reasoning === undefined
      ? []
      : [paragraph('Reasoning', heading), paragraph(message.reasoning, '<w:pStyle w:val="TranscriptReasoning"/>', '', state), paragraph('Reply', '<w:pStyle w:val="Heading2"/>')]),
    paragraph(message.text, toolResult ? '<w:pStyle w:val="TranscriptCode"/>' : '', '', state, toolResult),
    ...(message.attachments ?? []).map(a => attachment(a, state)),
    ...(activity ? [paragraph('Tool activity', heading)] : []),
    ...(message.tool_calls ?? []).flatMap((call, i) => [
      paragraph(`Tool: ${call.name}${call.id || call.call_id || call.uid ? ` · Call ${i + 1}` : ''}`, activity ? activityHeading : heading),
      paragraph(call.arguments, '<w:pStyle w:val="TranscriptCode"/>', '', state, true)
    ])
  ].join('')
}

export function renderDocx(doc: TranscriptDocument): string {
  const { associations, groups } = transcriptLayout(doc)

  const state: PackageState = {
    files: {},
    relationships: [],
    imageTypes: new Map(),
    nextId: 1,
    links: doc.options.include_links
  }

  const body = [
    paragraph(doc.title, '<w:pStyle w:val="Title"/>'),
    paragraph(`Conversation transcript · ${doc.messages.length} entries`, '<w:pStyle w:val="Metadata"/>'),
    paragraph(`Session: ${doc.session_id} · Exported: ${doc.exported_at}`, '<w:pStyle w:val="Metadata"/>'),
    ...(doc.model ? [paragraph(`Model: ${doc.model}`, '<w:pStyle w:val="Metadata"/>')] : []),
    ...doc.warnings.map(warning => paragraph(`Note: ${warning}`, '<w:pStyle w:val="Warning"/>')),
    ...groups.flatMap(({ index, results }) => [
      turn(doc.messages[index], state, doc.options.collapse_details, associations[index], false, results.length > 0),
      ...results.map(i => turn(doc.messages[i], state, doc.options.collapse_details, associations[i], true))
    ])
  ].join('')

  const imageTypes = [...state.imageTypes]
    .map(([extension, mime]) => `<Default Extension="${extension}" ContentType="${mime}"/>`)
    .join('')

  const parts: Record<string, string> = {
    'word/styles.xml': DOCX_STYLES,
    'word/settings.xml': DOCX_SETTINGS,
    'word/footer1.xml': DOCX_FOOTER,
    '[Content_Types].xml': `${XML}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>${imageTypes}<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>`,
    '_rels/.rels': `${XML}<Relationships xmlns="${PKG}"><Relationship Id="document" Type="${R}/officeDocument" Target="word/document.xml"/></Relationships>`,
    'word/_rels/document.xml.rels': `${XML}<Relationships xmlns="${PKG}"><Relationship Id="styles" Type="${R}/styles" Target="styles.xml"/><Relationship Id="settings" Type="${R}/settings" Target="settings.xml"/><Relationship Id="footer" Type="${R}/footer" Target="footer1.xml"/>${state.relationships.join('')}</Relationships>`,
    'word/document.xml': `${XML}<w:document xmlns:w="${W}" xmlns:r="${R}" xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" mc:Ignorable="w15" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>${body}<w:sectPr><w:footerReference w:type="default" r:id="footer"/><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>`
  }

  for (const [name, value] of Object.entries(parts)) {
    state.files[name] = strToU8(value)
  }

  return transcriptBase64(zipSync(state.files))
}
