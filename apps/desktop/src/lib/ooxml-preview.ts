export type OfficePreviewKind = 'document' | 'slides' | 'spreadsheet'

export type SpreadsheetCell = {
  align?: 'center' | 'left' | 'right'
  bold?: boolean
  color?: string
  fill?: string
  formula?: string
  italic?: boolean
  value: string
}

export type SpreadsheetSheet = {
  name: string
  rows: SpreadsheetCell[][]
}

export type SpreadsheetPreview = {
  kind: 'spreadsheet'
  sheets: SpreadsheetSheet[]
  truncated?: boolean
}

export type OfficeTextRun = {
  bold?: boolean
  italic?: boolean
  text: string
}

export type OfficeParagraph = {
  heading?: 1 | 2 | 3
  runs: OfficeTextRun[]
  type: 'paragraph'
}

export type OfficeTable = {
  rows: string[][]
  type: 'table'
}

export type OfficeBlock = OfficeParagraph | OfficeTable

export type DocumentPreview = {
  blocks: OfficeBlock[]
  kind: 'document'
  truncated?: boolean
}

export type SlidesPreview = {
  kind: 'slides'
  slides: { lines: string[] }[]
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

    return new Uint8Array(inflateRawSync(data, { maxOutputLength: MAX_PART_BYTES }))
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
    const reader = stream.getReader()
    const chunks: Uint8Array[] = []
    let total = 0

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        break
      }

      total += value.byteLength

      if (total > MAX_PART_BYTES) {
        await reader.cancel()

        return null
      }

      chunks.push(value)
    }

    const out = new Uint8Array(total)
    let offset = 0

    for (const chunk of chunks) {
      out.set(chunk, offset)
      offset += chunk.byteLength
    }

    return out
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

function parseDocx(zip: Map<string, Uint8Array>): DocumentPreview | null {
  const xml = zipText(zip, 'word/document.xml')
  const doc = xml ? parseXml(xml) : null
  const body = doc ? localElements(doc, 'body')[0] : null

  if (!body) {
    return null
  }

  const blocks = Array.from(body.children)
    .map(child => {
      if (child.localName === 'tbl') {
        return tableBlock(child)
      }

      if (child.localName === 'p') {
        return paragraphBlock(child)
      }

      return null
    })
    .filter((block): block is OfficeBlock => Boolean(block))

  return { blocks: blocks.length ? blocks : [{ runs: [], type: 'paragraph' }], kind: 'document' }
}

function paragraphBlock(paragraph: Element): OfficeParagraph {
  const style = paragraphStyle(paragraph)

  const heading: OfficeParagraph['heading'] =
    style === 'Heading1' || style === 'Title' ? 1 : style === 'Heading2' ? 2 : style === 'Heading3' ? 3 : undefined

  const runs = Array.from(paragraph.children)
    .filter(child => child.localName === 'r' || child.localName === 'hyperlink')
    .flatMap(child => (child.localName === 'hyperlink' ? collectRuns(child) : [textRun(child)]))
    .filter((run): run is OfficeTextRun => Boolean(run?.text))

  return heading ? { heading, runs, type: 'paragraph' } : { runs, type: 'paragraph' }
}

function paragraphStyle(paragraph: Element): string {
  const pPr = Array.from(paragraph.children).find(child => child.localName === 'pPr')

  if (!pPr) {
    return ''
  }

  const pStyle = localElements(pPr, 'pStyle')[0]

  return pStyle?.getAttribute('val') || pStyle?.getAttribute('w:val') || ''
}

function collectRuns(root: Element): OfficeTextRun[] {
  return Array.from(root.children)
    .filter(child => child.localName === 'r')
    .map(textRun)
    .filter((run): run is OfficeTextRun => Boolean(run?.text))
}

function textRun(run: Element): OfficeTextRun | null {
  const text = localElements(run, 't')
    .map(node => node.textContent || '')
    .join('')

  if (!text) {
    return null
  }

  const rPr = Array.from(run.children).find(child => child.localName === 'rPr')
  const next: OfficeTextRun = { text }

  if (rPr && localElements(rPr, 'b').length) {
    next.bold = true
  }

  if (rPr && localElements(rPr, 'i').length) {
    next.italic = true
  }

  return next
}

function tableBlock(table: Element): OfficeTable {
  const rows = localElements(table, 'tr').map(row =>
    localElements(row, 'tc').map(cell =>
      localElements(cell, 't')
        .map(node => node.textContent || '')
        .join('')
    )
  )

  return { rows, type: 'table' }
}

function parsePptx(zip: Map<string, Uint8Array>): SlidesPreview | null {
  const names = [...zip.keys()]
    .filter(name => /^ppt\/slides\/slide\d+\.xml$/i.test(name))
    .sort((a, b) => slideIndex(a) - slideIndex(b))

  if (!names.length) {
    return null
  }

  const truncated = names.length > MAX_SLIDES
  const slides = names.slice(0, MAX_SLIDES).map(name => ({ lines: slideLines(zipText(zip, name) || '') }))

  return { kind: 'slides', slides, truncated }
}

function slideIndex(name: string) {
  const match = /slide(\d+)\.xml$/i.exec(name)

  return match ? Number(match[1]) : 0
}

function slideLines(xml: string): string[] {
  const doc = parseXml(xml)

  if (!doc) {
    return []
  }

  return localElements(doc, 'p')
    .map(paragraph =>
      localElements(paragraph, 't')
        .map(node => node.textContent || '')
        .join('')
    )
    .filter(Boolean)
}

function parseXlsx(zip: Map<string, Uint8Array>): SpreadsheetPreview | null {
  const shared = parseSharedStrings(zipText(zip, 'xl/sharedStrings.xml'))
  const styles = parseStyles(zipText(zip, 'xl/styles.xml'))
  const date1904 = workbookUses1904Dates(zip)
  const sheets = listWorkbookSheets(zip)
  const truncated = sheets.length > MAX_SHEETS
  const selected = sheets.slice(0, MAX_SHEETS)

  const parsed = selected
    .map(sheet => {
      const xml = zipText(zip, sheet.path)

      if (!xml) {
        return null
      }

      const grid = parseSheetGrid(xml, shared, styles, date1904)

      return grid ? { name: sheet.name, rows: grid.rows, truncated: grid.truncated } : null
    })
    .filter((sheet): sheet is { name: string; rows: SpreadsheetCell[][]; truncated: boolean } => Boolean(sheet))

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

function parseSheetGrid(
  xml: string,
  shared: string[],
  styles: CellStyle[],
  date1904: boolean
): { rows: SpreadsheetCell[][]; truncated: boolean } | null {
  const doc = parseXml(xml)

  if (!doc) {
    return null
  }

  const cells: { col: number; row: number; cell: SpreadsheetCell }[] = []
  let truncated = false

  for (const node of localElements(doc, 'c')) {
    const ref = parseCellRef(node.getAttribute('r') || '')

    if (!ref) {
      continue
    }

    if (ref.row >= MAX_ROWS || ref.col >= MAX_COLS) {
      truncated = true

      continue
    }

    cells.push({ ...ref, cell: buildSheetCell(node, shared, styles, date1904) })
  }

  if (!cells.length) {
    return { rows: [], truncated }
  }

  const rowCount = Math.min(MAX_ROWS, Math.max(...cells.map(cell => cell.row)) + 1)
  const colCount = Math.min(MAX_COLS, Math.max(...cells.map(cell => cell.col)) + 1)
  const rows = Array.from({ length: rowCount }, () => Array.from({ length: colCount }, (): SpreadsheetCell => ({ value: '' })))

  for (const item of cells) {
    if (item.row < rowCount && item.col < colCount) {
      rows[item.row][item.col] = item.cell
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

type CellStyle = {
  align?: SpreadsheetCell['align']
  bold?: boolean
  color?: string
  fill?: string
  italic?: boolean
  numFmt?: string
}

const BUILTIN_NUM_FMTS: Record<number, string> = {
  0: 'General',
  1: '0',
  2: '0.00',
  3: '#,##0',
  4: '#,##0.00',
  9: '0%',
  10: '0.00%',
  14: 'm/d/yyyy',
  15: 'd-mmm-yy',
  16: 'd-mmm',
  17: 'mmm-yy',
  18: 'h:mm AM/PM',
  20: 'h:mm',
  21: 'h:mm:ss',
  22: 'm/d/yyyy h:mm',
  49: '@'
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function workbookUses1904Dates(zip: Map<string, Uint8Array>): boolean {
  const workbook = parseXml(zipText(zip, 'xl/workbook.xml') || '')
  const pr = workbook ? localElements(workbook, 'workbookPr')[0] : undefined
  const flag = pr?.getAttribute('date1904') || pr?.getAttribute('date1904') || ''

  return flag === '1' || flag === 'true'
}

function buildSheetCell(node: Element, shared: string[], styles: CellStyle[], date1904: boolean): SpreadsheetCell {
  const type = node.getAttribute('t') || ''
  const raw = cellValue(node, shared)

  const formula = Array.from(node.children)
    .find(child => child.localName === 'f')
    ?.textContent?.trim()

  const style = styles[Number(node.getAttribute('s') || '')] || {}
  const numeric = type === '' || type === 'n'
  const value = numeric ? formatExcelValue(raw, style.numFmt, date1904) : raw
  const cell: SpreadsheetCell = { value }

  if (formula) {
    cell.formula = formula.replace(/^\s*=/, '')
  }

  if (style.bold) {
    cell.bold = true
  }

  if (style.italic) {
    cell.italic = true
  }

  if (style.color) {
    cell.color = style.color
  }

  if (style.fill) {
    cell.fill = style.fill
  }

  if (style.align) {
    cell.align = style.align
  } else if (numeric && raw !== '' && !Number.isNaN(Number(raw))) {
    cell.align = 'right'
  }

  return cell
}

function parseStyles(xml: string | null): CellStyle[] {
  if (!xml) {
    return []
  }

  const doc = parseXml(xml)

  if (!doc) {
    return []
  }

  const numFmts = new Map<number, string>(Object.entries(BUILTIN_NUM_FMTS).map(([id, code]) => [Number(id), code]))

  for (const fmt of localElements(doc, 'numFmt')) {
    const id = Number(fmt.getAttribute('numFmtId'))
    const code = fmt.getAttribute('formatCode') || ''

    if (Number.isInteger(id) && code) {
      numFmts.set(id, code)
    }
  }

  const fonts = localElements(doc, 'font').map(font => ({
    bold: localElements(font, 'b').length > 0,
    italic: localElements(font, 'i').length > 0,
    color: rgbColor(localElements(font, 'color')[0])
  }))

  const fills = localElements(doc, 'fill').map(fill => {
    const pattern = localElements(fill, 'patternFill')[0]
    const type = pattern?.getAttribute('patternType') || ''

    if (type !== 'solid') {
      return undefined
    }

    return rgbColor(localElements(pattern, 'fgColor')[0])
  })

  return localElements(doc, 'xf')
    .filter(xf => xf.parentElement?.localName === 'cellXfs')
    .map(xf => {
      const style: CellStyle = {}
      const numFmtId = Number(xf.getAttribute('numFmtId') || '0')

      if (xf.getAttribute('applyNumberFormat') === '1') {
        style.numFmt = numFmts.get(numFmtId) || BUILTIN_NUM_FMTS[numFmtId]
      }

      if (xf.getAttribute('applyFont') === '1') {
        const font = fonts[Number(xf.getAttribute('fontId') || '0')]

        if (font?.bold) {
          style.bold = true
        }

        if (font?.italic) {
          style.italic = true
        }

        if (font?.color) {
          style.color = font.color
        }
      }

      if (xf.getAttribute('applyFill') === '1') {
        const fill = fills[Number(xf.getAttribute('fillId') || '0')]

        if (fill) {
          style.fill = fill
        }
      }

      if (xf.getAttribute('applyAlignment') === '1') {
        const alignment = localElements(xf, 'alignment')[0]?.getAttribute('horizontal')

        if (alignment === 'left' || alignment === 'center' || alignment === 'right') {
          style.align = alignment
        }
      }

      return style
    })
}

function rgbColor(node: Element | undefined): string | undefined {
  const rgb = node?.getAttribute('rgb') || ''
  const hex = rgb.replace(/^#/, '').slice(-6)

  return /^[0-9A-Fa-f]{6}$/.test(hex) ? `#${hex.toUpperCase()}` : undefined
}

function formatExcelValue(raw: string, numFmt: string | undefined, date1904: boolean): string {
  if (!raw || numFmt === undefined || numFmt === 'General' || numFmt === '@') {
    return raw
  }

  const number = Number(raw)

  if (!Number.isFinite(number)) {
    return raw
  }

  if (isDateFormat(numFmt)) {
    return formatExcelDate(number, numFmt, date1904)
  }

  if (numFmt.includes('%')) {
    const decimals = numFmt.includes('0.00') ? 2 : 0

    return `${(number * 100).toFixed(decimals)}%`
  }

  if (numFmt === '0') {
    return String(Math.round(number))
  }

  if (numFmt === '0.00' || numFmt === '#,##0.00') {
    const formatted = number.toFixed(2)

    return numFmt.includes('#,##') ? number.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : formatted
  }

  if (numFmt === '#,##0') {
    return Math.round(number).toLocaleString('en-US')
  }

  return raw
}

function isDateFormat(numFmt: string): boolean {
  const stripped = numFmt.replace(/"[^"]*"/g, '').replace(/\[[^\]]*]/g, '')

  return /[ymdhs]/i.test(stripped) && !stripped.includes('%')
}

function formatExcelDate(serial: number, numFmt: string, date1904: boolean): string {
  const whole = Math.floor(serial)
  const fraction = serial - whole
  const epoch = Date.UTC(date1904 ? 1904 : 1899, date1904 ? 0 : 11, date1904 ? 1 : 30)
  const date = new Date(epoch + whole * 86_400_000)
  const year = date.getUTCFullYear()
  const month = date.getUTCMonth() + 1
  const day = date.getUTCDate()
  const hours = Math.floor(fraction * 24)
  const minutes = Math.floor(fraction * 24 * 60) % 60
  const yy = String(year).slice(-2)
  const mon = MONTHS[month - 1] || ''
  const pad = (value: number) => String(value).padStart(2, '0')

  switch (numFmt) {
    case 'd-mmm-yy':
      return `${day}-${mon}-${yy}`

    case 'd-mmm':
      return `${day}-${mon}`

    case 'mmm-yy':
      return `${mon}-${yy}`

    case 'h:mm AM/PM':
      return formatClock(hours, minutes, true)

    case 'h:mm':
      return `${hours}:${pad(minutes)}`

    case 'h:mm:ss':
      return `${hours}:${pad(minutes)}:00`

    case 'm/d/yyyy h:mm':
      return `${month}/${day}/${year} ${hours}:${pad(minutes)}`

    default:
      return `${month}/${day}/${year}`
  }
}

function formatClock(hours: number, minutes: number, ampm: boolean): string {
  const pad = (value: number) => String(value).padStart(2, '0')

  if (!ampm) {
    return `${hours}:${pad(minutes)}`
  }

  const suffix = hours >= 12 ? 'PM' : 'AM'
  const hour12 = hours % 12 || 12

  return `${hour12}:${pad(minutes)} ${suffix}`
}
