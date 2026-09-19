import { evaluateFormula, type FormulaValue, formulaValueToRaw } from '@/lib/ooxml-formula'

export type OfficePreviewKind = 'document' | 'spreadsheet'

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
  color?: string
  fontFamily?: string
  fontSize?: number
  italic?: boolean
  text: string
  underline?: boolean
}

export type OfficeParagraph = {
  align?: 'center' | 'justify' | 'left' | 'right'
  dir?: 'ltr' | 'rtl'
  heading?: 1 | 2 | 3
  list?: 'bullet' | 'number'
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

export type OfficePreview = DocumentPreview | SpreadsheetPreview

const MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
const MAX_PART_BYTES = 8 * 1024 * 1024
const MAX_SHEETS = 32
const MAX_ROWS = 500
const MAX_COLS = 64

const NS_OFFICE_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

type ZipEntry = {
  compressedSize: number
  compression: number
  data: Uint8Array
  name: string
  uncompressedSize: number
}

export function isOfficePreviewKind(kind: string | undefined): kind is OfficePreviewKind {
  return kind === 'spreadsheet' || kind === 'document'
}

export function officePreviewKind(ext: string): OfficePreviewKind | null {
  switch (ext.toLowerCase()) {
    case '.xlsx':

    case '.xlsm':
      return 'spreadsheet'

    case '.docx':
      return 'document'

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

    return parseDocx(zip)
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
  const dir = paragraphDir(paragraph)

  const heading: OfficeParagraph['heading'] =
    style === 'Heading1' || style === 'Title' ? 1 : style === 'Heading2' ? 2 : style === 'Heading3' ? 3 : undefined

  const list: OfficeParagraph['list'] =
    style === 'ListBullet' || style === 'ListParagraph' ? 'bullet' : style === 'ListNumber' ? 'number' : undefined

  const align = paragraphAlign(paragraph, dir)

  const runs = Array.from(paragraph.children)
    .filter(child => child.localName === 'r' || child.localName === 'hyperlink')
    .flatMap(child => (child.localName === 'hyperlink' ? collectRuns(child) : [textRun(child)]))
    .filter((run): run is OfficeTextRun => Boolean(run?.text))

  return {
    ...(align ? { align } : {}),
    ...(dir ? { dir } : {}),
    ...(heading ? { heading } : {}),
    ...(list ? { list } : {}),
    runs,
    type: 'paragraph'
  }
}

function paragraphDir(paragraph: Element): OfficeParagraph['dir'] | undefined {
  const pPr = Array.from(paragraph.children).find(child => child.localName === 'pPr')

  if (!pPr) {
    return undefined
  }

  const mark = localElements(pPr, 'bidi')[0] || localElements(pPr, 'rtl')[0]

  return isOn(mark) ? 'rtl' : undefined
}

function paragraphAlign(paragraph: Element, dir?: OfficeParagraph['dir']): OfficeParagraph['align'] | undefined {
  const pPr = Array.from(paragraph.children).find(child => child.localName === 'pPr')
  const jc = pPr ? localElements(pPr, 'jc')[0] : undefined
  const value = (wordVal(jc) || '').toLowerCase()

  if (value === 'center' || value === 'left' || value === 'right') {
    return value
  }

  if (value === 'both') {
    return 'justify'
  }

  if (value === 'start') {
    return dir === 'rtl' ? 'right' : 'left'
  }

  if (value === 'end') {
    return dir === 'rtl' ? 'left' : 'right'
  }

  return undefined
}

function wordVal(node: Element | undefined): string {
  return node?.getAttribute('val') || node?.getAttribute('w:val') || ''
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

  if (!rPr) {
    return next
  }

  if (isOn(localElements(rPr, 'b')[0])) {
    next.bold = true
  }

  if (isOn(localElements(rPr, 'i')[0])) {
    next.italic = true
  }

  const underline = localElements(rPr, 'u')[0]

  if (underline) {
    const value = (wordVal(underline) || 'single').toLowerCase()

    if (value !== 'none' && value !== '0' && value !== 'false') {
      next.underline = true
    }
  }

  const color = wordVal(localElements(rPr, 'color')[0])

  if (color && color.toLowerCase() !== 'auto') {
    next.color = `#${color.replace(/^FF/i, '').slice(-6)}`
  }

  const size = Number(wordVal(localElements(rPr, 'sz')[0]))

  if (size > 0) {
    next.fontSize = size / 2
  }

  const fonts = localElements(rPr, 'rFonts')[0]
  const family = wordFontFamily(fonts)

  const theme =
    fonts?.getAttribute('asciiTheme') ||
    fonts?.getAttribute('hAnsiTheme') ||
    fonts?.getAttribute('eastAsiaTheme') ||
    fonts?.getAttribute('cstheme') ||
    fonts?.getAttribute('w:asciiTheme')

  if (family) {
    next.fontFamily = family
  } else if (theme) {
    next.fontFamily = /major/i.test(theme) ? 'Cambria' : 'Calibri'
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

function parseXlsx(zip: Map<string, Uint8Array>): SpreadsheetPreview | null {
  const shared = parseSharedStrings(zipText(zip, 'xl/sharedStrings.xml'))
  const styles = parseStyles(zipText(zip, 'xl/styles.xml'))
  const date1904 = workbookUses1904Dates(zip)
  const sheets = listWorkbookSheets(zip)
  const truncated = sheets.length > MAX_SHEETS
  const selected = sheets.slice(0, MAX_SHEETS)

  const drafts = selected
    .map(sheet => {
      const xml = zipText(zip, sheet.path)

      if (!xml) {
        return null
      }

      const parsed = parseSheetDrafts(xml, shared, styles)

      return parsed ? { name: sheet.name, ...parsed } : null
    })
    .filter((sheet): sheet is { name: string; cells: DraftCell[]; truncated: boolean } => Boolean(sheet))

  if (!drafts.length) {
    return null
  }

  evaluateDrafts(drafts)

  return {
    kind: 'spreadsheet',
    sheets: drafts.map(sheet => ({
      name: sheet.name,
      rows: draftsToRows(sheet.cells, date1904)
    })),
    truncated: truncated || drafts.some(sheet => sheet.truncated)
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

type DraftCell = {
  col: number
  formula?: string
  raw: string
  row: number
  style: CellStyle
  type: string
}

function parseSheetDrafts(
  xml: string,
  shared: string[],
  styles: CellStyle[]
): { cells: DraftCell[]; truncated: boolean } | null {
  const doc = parseXml(xml)

  if (!doc) {
    return null
  }

  const cells: DraftCell[] = []
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

    const type = node.getAttribute('t') || ''

    const formula = Array.from(node.children)
      .find(child => child.localName === 'f')
      ?.textContent?.trim()

    cells.push({
      ...ref,
      formula: formula || undefined,
      raw: cellValue(node, shared),
      style: styles[Number(node.getAttribute('s') || '')] || {},
      type
    })
  }

  return { cells, truncated }
}

function evaluateDrafts(sheets: { cells: DraftCell[]; name: string }[]) {
  const index = new Map<string, DraftCell>()

  for (const sheet of sheets) {
    for (const cell of sheet.cells) {
      index.set(`${sheet.name}!${cell.col}:${cell.row}`, cell)
    }
  }

  const visiting = new Set<string>()
  const computed = new Map<string, FormulaValue>()

  const get = (sheet: string, col: number, row: number): FormulaValue => {
    const key = `${sheet}!${col}:${row}`

    if (computed.has(key)) {
      return computed.get(key) as FormulaValue
    }

    const cell = index.get(key)

    if (!cell) {
      return ''
    }

    if (cell.formula && cell.raw === '') {
      if (visiting.has(key)) {
        computed.set(key, '#REF!')

        return '#REF!'
      }

      visiting.add(key)

      try {
        const value = evaluateFormula(cell.formula, { currentSheet: sheet, get })
        computed.set(key, value)
        cell.raw = formulaValueToRaw(value)

        if (typeof value === 'number') {
          cell.type = 'n'
        }

        return value
      } catch {
        computed.set(key, '#VALUE!')
        cell.raw = '#VALUE!'

        return '#VALUE!'
      } finally {
        visiting.delete(key)
      }
    }

    const primitive = primitiveFromDraft(cell)
    computed.set(key, primitive)

    return primitive
  }

  for (const sheet of sheets) {
    for (const cell of sheet.cells) {
      if (cell.formula && cell.raw === '') {
        get(sheet.name, cell.col, cell.row)
      }
    }
  }
}

function primitiveFromDraft(cell: DraftCell): FormulaValue {
  if (cell.type === 'b') {
    return cell.raw === 'TRUE' || cell.raw === '1'
  }

  if (cell.raw === '') {
    return ''
  }

  if (cell.type === '' || cell.type === 'n') {
    const number = Number(cell.raw)

    return Number.isFinite(number) ? number : cell.raw
  }

  return cell.raw
}

function draftsToRows(cells: DraftCell[], date1904: boolean): SpreadsheetCell[][] {
  if (!cells.length) {
    return []
  }

  const rowCount = Math.min(MAX_ROWS, Math.max(...cells.map(cell => cell.row)) + 1)
  const colCount = Math.min(MAX_COLS, Math.max(...cells.map(cell => cell.col)) + 1)
  const rows = Array.from({ length: rowCount }, () => Array.from({ length: colCount }, (): SpreadsheetCell => ({ value: '' })))

  for (const draft of cells) {
    if (draft.row < rowCount && draft.col < colCount) {
      rows[draft.row][draft.col] = displayCell(draft, date1904)
    }
  }

  return rows
}

function displayCell(draft: DraftCell, date1904: boolean): SpreadsheetCell {
  const numeric = draft.type === '' || draft.type === 'n'
  const value = numeric ? formatExcelValue(draft.raw, draft.style.numFmt, date1904) : draft.raw
  const cell: SpreadsheetCell = { value }

  if (draft.formula) {
    cell.formula = draft.formula.replace(/^\s*=/, '')
  }

  if (draft.style.bold) {cell.bold = true}

  if (draft.style.italic) {cell.italic = true}

  if (draft.style.color) {cell.color = draft.style.color}

  if (draft.style.fill) {cell.fill = draft.style.fill}

  if (draft.style.align) {cell.align = draft.style.align}
  else if (numeric && draft.raw !== '' && !Number.isNaN(Number(draft.raw))) {cell.align = 'right'}

  return cell
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

  const fonts = localElements(doc, 'fonts')[0]
    ? Array.from(localElements(doc, 'fonts')[0].children)
        .filter(node => node.localName === 'font')
        .map(font => ({
          bold: isOn(localElements(font, 'b')[0]),
          italic: isOn(localElements(font, 'i')[0]),
          color: rgbColor(localElements(font, 'color')[0])
        }))
    : []

  const fills = localElements(doc, 'fills')[0]
    ? Array.from(localElements(doc, 'fills')[0].children)
        .filter(node => node.localName === 'fill')
        .map(fill => {
          const pattern = localElements(fill, 'patternFill')[0]
          const type = pattern?.getAttribute('patternType') || ''

          if (type !== 'solid') {
            return undefined
          }

          return rgbColor(localElements(pattern, 'fgColor')[0])
        })
    : []

  const cellXfs = localElements(doc, 'cellXfs')[0]
  const xfs = cellXfs ? Array.from(cellXfs.children).filter(node => node.localName === 'xf') : []

  return xfs.map(xf => {
    const style: CellStyle = {}
    const numFmtId = Number(xf.getAttribute('numFmtId') || '0')

    if (xf.getAttribute('applyNumberFormat') !== '0') {
      style.numFmt = numFmts.get(numFmtId) || BUILTIN_NUM_FMTS[numFmtId]
    }

    if (xf.getAttribute('applyFont') !== '0') {
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

    if (xf.getAttribute('applyFill') !== '0') {
      const fill = fills[Number(xf.getAttribute('fillId') || '0')]

      if (fill) {
        style.fill = fill
      }
    }

    if (xf.getAttribute('applyAlignment') !== '0') {
      const alignment = localElements(xf, 'alignment')[0]?.getAttribute('horizontal')

      if (alignment === 'left' || alignment === 'center' || alignment === 'right') {
        style.align = alignment
      }
    }

    return style
  })
}

function wordFontFamily(fonts: Element | undefined): string | undefined {
  if (!fonts) {
    return undefined
  }

  const names = ['eastAsia', 'cs', 'ascii', 'hAnsi']
    .map(key => fonts.getAttribute(key) || fonts.getAttribute(`w:${key}`) || '')
    .map(name => name.trim())
    .filter(Boolean)

  return names.length ? [...new Set(names)].join(', ') : undefined
}

function isOn(node: Element | undefined): boolean {
  if (!node) {
    return false
  }

  const value = (node.getAttribute('val') || node.getAttribute('w:val') || '1').toLowerCase()

  return value !== '0' && value !== 'false' && value !== 'off'
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

  const code = numFmt.split(';')[0] || numFmt

  if (isDateFormat(code)) {
    return formatExcelDate(number, code, date1904)
  }

  if (code.includes('%')) {
    const decimals = /\.0+/.exec(code)?.[0].length - 1 || 0

    return `${(number * 100).toFixed(decimals)}%`
  }

  if (code.includes('$') && /#,##0/.test(code)) {
    const decimals = code.includes('0.00') ? 2 : 0

    const formatted = Math.abs(number).toLocaleString('en-US', {
      maximumFractionDigits: decimals,
      minimumFractionDigits: decimals
    })

    return number < 0 ? `($${formatted})` : `$${formatted}`
  }

  if (code === '0') {
    return String(Math.round(number))
  }

  if (code === '0.00' || code === '#,##0.00') {
    const formatted = number.toFixed(2)

    return code.includes('#,##')
      ? number.toLocaleString('en-US', { maximumFractionDigits: 2, minimumFractionDigits: 2 })
      : formatted
  }

  if (code === '#,##0') {
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

    case 'mm/dd/yyyy':
      return `${pad(month)}/${pad(day)}/${year}`

    case 'yyyy-mm-dd h:mm:ss':
      return `${year}-${pad(month)}-${pad(day)} ${pad(hours)}:${pad(minutes)}:00`

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
