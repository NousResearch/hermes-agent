export type OfficePreviewKind = 'document' | 'slides' | 'spreadsheet'

export type SpreadsheetSheet = {
  name: string
  rows: string[][]
}

export type SpreadsheetPreview = {
  kind: 'spreadsheet'
  sheets: SpreadsheetSheet[]
  truncated?: boolean
}

export type DocumentPreview = {
  html: string
  kind: 'document'
  truncated?: boolean
}

export type SlidesPreview = {
  kind: 'slides'
  slides: { html: string }[]
  truncated?: boolean
}

export type OfficePreview = DocumentPreview | SlidesPreview | SpreadsheetPreview

const MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
const MAX_PART_BYTES = 8 * 1024 * 1024
const MAX_SHEETS = 32
const MAX_ROWS = 500
const MAX_COLS = 64
const MAX_SLIDES = 50

const NS_OFFICE_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

type ZipEntry = {
  compressedSize: number
  compression: number
  data: Uint8Array
  name: string
  uncompressedSize: number
}

export function isOfficePreviewKind(kind: string | undefined): kind is OfficePreviewKind {
  return kind === 'spreadsheet' || kind === 'document' || kind === 'slides'
}

export function officePreviewKind(ext: string): OfficePreviewKind | null {
  switch (ext.toLowerCase()) {
    case '.xlsx':

    case '.xlsm':
      return 'spreadsheet'

    case '.docx':
      return 'document'

    case '.pptx':
      return 'slides'

    default:
      return null
  }
}

export async function parseOfficePreview(bytes: Uint8Array, ext: string): Promise<OfficePreview | null> {
  const kind = officePreviewKind(ext)

  if (!kind || bytes.byteLength < 22 || bytes.byteLength > MAX_ARCHIVE_BYTES) {
    return null
  }

  if (bytes[0] !== 0x50 || bytes[1] !== 0x4b) {
    return null
  }

  try {
    const zip = await readZip(bytes)

    if (kind === 'spreadsheet') {
      return parseXlsx(zip)
    }

    if (kind === 'document') {
      return parseDocx(zip)
    }

    return parsePptx(zip)
  } catch {
    return null
  }
}

async function readZip(bytes: Uint8Array): Promise<Map<string, Uint8Array>> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const eocd = findEocd(bytes)

  if (eocd < 0) {
    throw new Error('missing zip eocd')
  }

  const entries = view.getUint16(eocd + 10, true)
  const cdSize = view.getUint32(eocd + 12, true)
  const cdOffset = view.getUint32(eocd + 16, true)

  if (cdOffset + cdSize > bytes.byteLength) {
    throw new Error('truncated zip directory')
  }

  const files = new Map<string, Uint8Array>()
  let cursor = cdOffset

  for (let index = 0; index < entries; index += 1) {
    if (cursor + 46 > bytes.byteLength || view.getUint32(cursor, true) !== 0x02014b50) {
      break
    }

    const compression = view.getUint16(cursor + 10, true)
    const compressedSize = view.getUint32(cursor + 20, true)
    const uncompressedSize = view.getUint32(cursor + 24, true)
    const nameLen = view.getUint16(cursor + 28, true)
    const extraLen = view.getUint16(cursor + 30, true)
    const commentLen = view.getUint16(cursor + 32, true)
    const localOffset = view.getUint32(cursor + 42, true)
    const name = decodeUtf8(bytes.subarray(cursor + 46, cursor + 46 + nameLen))
    cursor += 46 + nameLen + extraLen + commentLen

    if (!name || name.endsWith('/') || uncompressedSize > MAX_PART_BYTES || compressedSize > MAX_PART_BYTES) {
      continue
    }

    const entry: ZipEntry = {
      compressedSize,
      compression,
      data: bytes,
      name,
      uncompressedSize
    }

    const payload = await readZipPayload(view, bytes, localOffset, entry)

    if (payload) {
      files.set(name.replace(/^\/+/, ''), payload)
    }
  }

  return files
}

function findEocd(bytes: Uint8Array): number {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const min = Math.max(0, bytes.byteLength - 22 - 0xffff)

  for (let offset = bytes.byteLength - 22; offset >= min; offset -= 1) {
    if (view.getUint32(offset, true) === 0x06054b50) {
      const commentLen = view.getUint16(offset + 20, true)

      if (offset + 22 + commentLen === bytes.byteLength) {
        return offset
      }
    }
  }

  return -1
}

async function readZipPayload(
  view: DataView,
  bytes: Uint8Array,
  localOffset: number,
  entry: ZipEntry
): Promise<Uint8Array | null> {
  if (localOffset + 30 > bytes.byteLength || view.getUint32(localOffset, true) !== 0x04034b50) {
    return null
  }

  const nameLen = view.getUint16(localOffset + 26, true)
  const extraLen = view.getUint16(localOffset + 28, true)
  const dataStart = localOffset + 30 + nameLen + extraLen
  const dataEnd = dataStart + entry.compressedSize

  if (dataEnd > bytes.byteLength) {
    return null
  }

  const payload = bytes.subarray(dataStart, dataEnd)

  if (entry.compression === 0) {
    return new Uint8Array(payload)
  }

  if (entry.compression !== 8) {
    return null
  }

  return inflateRaw(payload)
}

async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
  const inflated = (await inflateRawWithDecompressionStream(data)) || (await inflateRawWithZlib(data))

  if (!inflated) {
    throw new Error('deflate-raw is unavailable')
  }

  if (inflated.byteLength > MAX_PART_BYTES) {
    throw new Error('inflated part too large')
  }

  return inflated
}

async function inflateRawWithZlib(data: Uint8Array): Promise<Uint8Array | null> {
  try {
    const { inflateRawSync } = await import(/* @vite-ignore */ 'node:zlib')

    return new Uint8Array(inflateRawSync(data))
  } catch {
    return null
  }
}

async function inflateRawWithDecompressionStream(data: Uint8Array): Promise<Uint8Array | null> {
  if (typeof DecompressionStream !== 'function') {
    return null
  }

  try {
    const copy = new Uint8Array(data)
    const stream = new Blob([copy]).stream().pipeThrough(new DecompressionStream('deflate-raw'))

    return new Uint8Array(await new Response(stream).arrayBuffer())
  } catch {
    return null
  }
}

function decodeUtf8(bytes: Uint8Array) {
  return new TextDecoder('utf-8', { fatal: false }).decode(bytes)
}

function parseXml(text: string): Document | null {
  const doc = new DOMParser().parseFromString(text, 'application/xml')

  if (doc.getElementsByTagName('parsererror').length > 0) {
    return null
  }

  return doc
}

function zipText(zip: Map<string, Uint8Array>, name: string): string | null {
  const bytes = zip.get(name)

  return bytes ? decodeUtf8(bytes) : null
}

function localElements(root: ParentNode, name: string): Element[] {
  return Array.from(root.querySelectorAll('*')).filter(el => el.localName === name)
}

function attr(el: Element, name: string): string {
  return el.getAttribute(name) || el.getAttributeNS(NS_OFFICE_REL, name) || ''
}

function escapeHtml(value: string) {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

function parseXlsx(zip: Map<string, Uint8Array>): SpreadsheetPreview | null {
  const shared = parseSharedStrings(zipText(zip, 'xl/sharedStrings.xml'))
  const sheets = listWorkbookSheets(zip)
  const truncated = sheets.length > MAX_SHEETS
  const selected = sheets.slice(0, MAX_SHEETS)

  const parsed = selected
    .map(sheet => {
      const xml = zipText(zip, sheet.path)

      if (!xml) {
        return null
      }

      const grid = parseSheetGrid(xml, shared)

      return grid ? { name: sheet.name, rows: grid.rows, truncated: grid.truncated } : null
    })
    .filter((sheet): sheet is { name: string; rows: string[][]; truncated: boolean } => Boolean(sheet))

  if (!parsed.length) {
    return null
  }

  return {
    kind: 'spreadsheet',
    sheets: parsed.map(({ name, rows }) => ({ name, rows })),
    truncated: truncated || parsed.some(sheet => sheet.truncated)
  }
}

function listWorkbookSheets(zip: Map<string, Uint8Array>): { name: string; path: string }[] {
  const workbook = parseXml(zipText(zip, 'xl/workbook.xml') || '')
  const rels = parseWorkbookRels(zipText(zip, 'xl/_rels/workbook.xml.rels') || '')

  if (!workbook) {
    return fallbackSheets(zip)
  }

  const listed = localElements(workbook, 'sheet')
    .map((sheet, index) => {
      const name = sheet.getAttribute('name') || `Sheet ${index + 1}`
      const rid = attr(sheet, 'id')
      const target = rid ? rels.get(rid) : undefined
      const path = target ? resolveWorkbookTarget(target) : `xl/worksheets/sheet${index + 1}.xml`

      return zip.has(path) ? { name, path } : null
    })
    .filter((sheet): sheet is { name: string; path: string } => Boolean(sheet))

  return listed.length ? listed : fallbackSheets(zip)
}

function fallbackSheets(zip: Map<string, Uint8Array>) {
  return [...zip.keys()]
    .filter(name => /^xl\/worksheets\/sheet\d+\.xml$/i.test(name))
    .sort((a, b) => sheetIndex(a) - sheetIndex(b))
    .map((path, index) => ({ name: `Sheet ${index + 1}`, path }))
}

function sheetIndex(name: string) {
  const match = /sheet(\d+)\.xml$/i.exec(name)

  return match ? Number(match[1]) : 0
}

function resolveWorkbookTarget(target: string) {
  const normalized = target.replace(/\\/g, '/').replace(/^\.\//, '')

  if (normalized.startsWith('/')) {
    return normalized.slice(1)
  }

  if (normalized.startsWith('xl/')) {
    return normalized
  }

  return `xl/${normalized}`
}

function parseWorkbookRels(xml: string) {
  const doc = parseXml(xml)
  const rels = new Map<string, string>()

  if (!doc) {
    return rels
  }

  for (const rel of localElements(doc, 'Relationship')) {
    const id = rel.getAttribute('Id')
    const target = rel.getAttribute('Target')

    if (id && target) {
      rels.set(id, target)
    }
  }

  return rels
}

function parseSharedStrings(xml: string | null): string[] {
  if (!xml) {
    return []
  }

  const doc = parseXml(xml)

  if (!doc) {
    return []
  }

  return localElements(doc, 'si').map(si =>
    localElements(si, 't')
      .map(node => node.textContent || '')
      .join('')
  )
}

function parseSheetGrid(xml: string, shared: string[]): { rows: string[][]; truncated: boolean } | null {
  const doc = parseXml(xml)

  if (!doc) {
    return null
  }

  const cells: { col: number; row: number; value: string }[] = []
  let truncated = false

  for (const cell of localElements(doc, 'c')) {
    const ref = parseCellRef(cell.getAttribute('r') || '')

    if (!ref) {
      continue
    }

    if (ref.row >= MAX_ROWS || ref.col >= MAX_COLS) {
      truncated = true

      continue
    }

    cells.push({ ...ref, value: cellValue(cell, shared) })
  }

  if (!cells.length) {
    return { rows: [], truncated }
  }

  const rowCount = Math.min(MAX_ROWS, Math.max(...cells.map(cell => cell.row)) + 1)
  const colCount = Math.min(MAX_COLS, Math.max(...cells.map(cell => cell.col)) + 1)
  const rows = Array.from({ length: rowCount }, () => Array.from({ length: colCount }, () => ''))

  for (const cell of cells) {
    if (cell.row < rowCount && cell.col < colCount) {
      rows[cell.row][cell.col] = cell.value
    }
  }

  return { rows, truncated }
}

function parseCellRef(ref: string): { col: number; row: number } | null {
  const match = /^([A-Z]+)(\d+)$/i.exec(ref.trim())

  if (!match) {
    return null
  }

  let col = 0

  for (const char of match[1].toUpperCase()) {
    col = col * 26 + (char.charCodeAt(0) - 64)
  }

  return { col: col - 1, row: Number(match[2]) - 1 }
}

function cellValue(cell: Element, shared: string[]): string {
  const type = cell.getAttribute('t') || ''

  if (type === 's') {
    const index = Number(directValue(cell))

    return Number.isInteger(index) ? shared[index] || '' : ''
  }

  if (type === 'inlineStr') {
    return localElements(cell, 't')
      .map(node => node.textContent || '')
      .join('')
  }

  if (type === 'b') {
    return directValue(cell) === '1' ? 'TRUE' : 'FALSE'
  }

  return directValue(cell)
}

function directValue(cell: Element): string {
  const value = Array.from(cell.children).find(child => child.localName === 'v')

  return (value?.textContent || '').trim()
}

function parseDocx(zip: Map<string, Uint8Array>): DocumentPreview | null {
  const xml = zipText(zip, 'word/document.xml')
  const doc = xml ? parseXml(xml) : null
  const body = doc ? localElements(doc, 'body')[0] : null

  if (!body) {
    return null
  }

  const html = Array.from(body.children)
    .map(child => {
      if (child.localName === 'tbl') {
        return renderTable(child)
      }

      if (child.localName === 'p') {
        return renderParagraph(child)
      }

      return ''
    })
    .join('')

  return { kind: 'document', html: html || '<p></p>' }
}

function renderParagraph(paragraph: Element): string {
  const style = paragraphStyle(paragraph)
  const tag =
    style === 'Heading1' || style === 'Title' ? 'h1' : style === 'Heading2' ? 'h2' : style === 'Heading3' ? 'h3' : 'p'

  const inner = Array.from(paragraph.children)
    .filter(child => child.localName === 'r' || child.localName === 'hyperlink')
    .map(child => (child.localName === 'hyperlink' ? renderRuns(child) : renderRun(child)))
    .join('')

  return `<${tag}>${inner || '<br>'}</${tag}>`
}

function paragraphStyle(paragraph: Element): string {
  const pPr = Array.from(paragraph.children).find(child => child.localName === 'pPr')

  if (!pPr) {
    return ''
  }

  const pStyle = localElements(pPr, 'pStyle')[0]

  return pStyle?.getAttribute('val') || pStyle?.getAttribute('w:val') || ''
}

function renderRuns(root: Element): string {
  return Array.from(root.children)
    .filter(child => child.localName === 'r')
    .map(renderRun)
    .join('')
}

function renderRun(run: Element): string {
  const text = localElements(run, 't')
    .map(node => node.textContent || '')
    .join('')

  if (!text) {
    return ''
  }

  const rPr = Array.from(run.children).find(child => child.localName === 'rPr')
  const bold = Boolean(rPr && localElements(rPr, 'b').length)
  const italic = Boolean(rPr && localElements(rPr, 'i').length)
  let html = escapeHtml(text)

  if (bold) {
    html = `<strong>${html}</strong>`
  }

  if (italic) {
    html = `<em>${html}</em>`
  }

  return html
}

function renderTable(table: Element): string {
  const rows = localElements(table, 'tr')
    .map(row => {
      const cells = localElements(row, 'tc')
        .map(cell => {
          const text = localElements(cell, 't')
            .map(node => escapeHtml(node.textContent || ''))
            .join('')

          return `<td>${text}</td>`
        })
        .join('')

      return `<tr>${cells}</tr>`
    })
    .join('')

  return `<table>${rows}</table>`
}

function parsePptx(zip: Map<string, Uint8Array>): SlidesPreview | null {
  const names = [...zip.keys()]
    .filter(name => /^ppt\/slides\/slide\d+\.xml$/i.test(name))
    .sort((a, b) => slideIndex(a) - slideIndex(b))

  if (!names.length) {
    return null
  }

  const truncated = names.length > MAX_SLIDES
  const slides = names.slice(0, MAX_SLIDES).map(name => ({ html: renderSlide(zipText(zip, name) || '') }))

  return { kind: 'slides', slides, truncated }
}

function slideIndex(name: string) {
  const match = /slide(\d+)\.xml$/i.exec(name)

  return match ? Number(match[1]) : 0
}

function renderSlide(xml: string): string {
  const doc = parseXml(xml)

  if (!doc) {
    return '<p></p>'
  }

  const paragraphs = localElements(doc, 'p')
    .map(paragraph => {
      const text = localElements(paragraph, 't')
        .map(node => node.textContent || '')
        .join('')

      return text ? `<p>${escapeHtml(text)}</p>` : ''
    })
    .filter(Boolean)
    .join('')

  return paragraphs || '<p></p>'
}
