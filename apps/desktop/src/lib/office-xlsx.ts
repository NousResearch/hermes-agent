import { unzipSync } from 'fflate'

export interface SheetPreview {
  name: string
  rows: string[][]
  /** Excel character-width converted to pixels, one entry per used column. */
  widths?: number[]
  truncated: boolean
}

const MAX_ROWS = 200
const MAX_COLS = 40
const MAX_UNZIPPED_BYTES = 32 * 1024 * 1024

function textOf(bytes: Uint8Array | undefined): string {
  return bytes ? new TextDecoder().decode(bytes) : ''
}

function localName(element: Element): string {
  return element.localName
}

function attribute(element: Element, name: string): string | null {
  for (const attr of element.attributes) {
    if (attr.localName === name) {
      return attr.value
    }
  }

  return null
}

function descendants(root: ParentNode, name: string): Element[] {
  return Array.from(root.querySelectorAll('*')).filter(element => localName(element) === name)
}

function parseXml(xml: string): Document {
  const document = new DOMParser().parseFromString(xml, 'application/xml')

  if (document.querySelector('parsererror')) {
    throw new Error('Could not read this spreadsheet')
  }

  return document
}

function columnIndex(cellRef: string): number {
  const letters = /^[A-Z]+/i.exec(cellRef)?.[0] ?? ''
  let index = 0

  for (const char of letters.toUpperCase()) {
    index = index * 26 + (char.charCodeAt(0) - 64)
  }

  return Math.max(0, index - 1)
}

function sharedStrings(bytes: Uint8Array | undefined): string[] {
  if (!bytes?.length) {
    return []
  }

  return descendants(parseXml(textOf(bytes)), 'si').map(item =>
    descendants(item, 't')
      .map(node => node.textContent ?? '')
      .join('')
  )
}

const DATE_FORMAT_IDS = new Set([14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47])

function isDateFormat(id: number, code: string | undefined): boolean {
  if (DATE_FORMAT_IDS.has(id)) {
    return true
  }

  if (!code) {
    return false
  }

  const stripped = code.replace(/\[[^\]]*\]/g, '')

  return /[ydmhs]/i.test(stripped) && !/^[#0?,.\s%]+$/.test(stripped)
}

function excelDate(serial: number, withTime: boolean): string {
  const date = new Date(Date.UTC(1899, 11, 30) + Math.round(serial * 86400000))

  if (Number.isNaN(date.getTime())) {
    return String(serial)
  }

  const day = String(date.getUTCDate()).padStart(2, '0')
  const month = String(date.getUTCMonth() + 1).padStart(2, '0')
  const stamp = `${date.getUTCFullYear()}-${month}-${day}`

  if (!withTime) {
    return stamp
  }

  const hours = String(date.getUTCHours()).padStart(2, '0')
  const minutes = String(date.getUTCMinutes()).padStart(2, '0')

  return `${stamp} ${hours}:${minutes}`
}

function formatNumber(value: string, formatId: number, formatCode: string | undefined): string {
  const number = Number(value)

  if (!value || Number.isNaN(number)) {
    return value
  }

  if (formatId === 9 || formatId === 10 || formatCode?.includes('%')) {
    return `${formatId === 10 ? (number * 100).toFixed(2) : Math.round(number * 100)}%`
  }

  if (isDateFormat(formatId, formatCode)) {
    return excelDate(number, formatId === 22 || /h/i.test(formatCode || ''))
  }

  return value
}

function styleFormats(bytes: Uint8Array | undefined): { ids: number[]; codes: Map<number, string> } {
  if (!bytes?.length) {
    return { ids: [], codes: new Map() }
  }

  const document = parseXml(textOf(bytes))
  const codes = new Map<number, string>()

  for (const format of descendants(document, 'numFmt')) {
    const id = Number(attribute(format, 'numFmtId'))
    const code = attribute(format, 'formatCode')

    if (code) {
      codes.set(id, code)
    }
  }

  const ids = descendants(document, 'xf').map(xf => Number(attribute(xf, 'numFmtId') || 0))
  const cellXfs = descendants(document, 'cellXfs')[0]
  const used = cellXfs ? descendants(cellXfs, 'xf').map(xf => Number(attribute(xf, 'numFmtId') || 0)) : ids

  return { ids: used, codes }
}

function cellText(cell: Element, strings: string[], formats: { ids: number[]; codes: Map<number, string> }): string {
  const type = cell.getAttribute('t')
  const value = descendants(cell, 'v')[0]?.textContent ?? ''

  const inline = descendants(cell, 't')
    .map(node => node.textContent ?? '')
    .join('')

  if (type === 's') {
    return strings[Number(value)] ?? ''
  }

  if (type === 'inlineStr' || type === 'str') {
    return inline || value
  }

  if (type === 'b') {
    return value === '1' ? 'TRUE' : 'FALSE'
  }

  const style = Number(cell.getAttribute('s') || 0)
  const formatId = formats.ids[style] ?? 0

  return formatNumber(value || inline, formatId, formats.codes.get(formatId))
}

function columnWidths(document: Document, colCount: number): number[] {
  const widths = Array.from({ length: Math.max(colCount, 1) }, () => 88)

  for (const col of descendants(document, 'col')) {
    const min = Number(attribute(col, 'min') || 1)
    const max = Number(attribute(col, 'max') || min)
    const width = Number(attribute(col, 'width') || 0)

    if (!width) {
      continue
    }

    const pixels = Math.max(32, Math.round(width * 8))

    for (let index = min - 1; index < Math.min(max, widths.length); index += 1) {
      widths[index] = pixels
    }
  }

  return widths
}

function sheetGrid(
  bytes: Uint8Array,
  strings: string[],
  formats: { ids: number[]; codes: Map<number, string> }
): { rows: string[][]; widths: number[]; truncated: boolean } {
  const document = parseXml(textOf(bytes))
  const cells = descendants(document, 'c')
  const grid = new Map<number, Map<number, string>>()
  let maxRow = -1
  let maxCol = -1
  let truncated = false

  for (const cell of cells) {
    const ref = cell.getAttribute('r') ?? ''
    const row = Math.max(0, (Number(cell.parentElement?.getAttribute('r')) || 1) - 1)
    const col = columnIndex(ref)

    if (row >= MAX_ROWS || col >= MAX_COLS) {
      truncated = true

      continue
    }

    const rowCells = grid.get(row) ?? new Map<number, string>()
    rowCells.set(col, cellText(cell, strings, formats))
    grid.set(row, rowCells)
    maxRow = Math.max(maxRow, row)
    maxCol = Math.max(maxCol, col)
  }

  const rows: string[][] = []

  for (let row = 0; row <= maxRow; row += 1) {
    const source = grid.get(row)
    rows.push(Array.from({ length: maxCol + 1 }, (_, col) => source?.get(col) ?? ''))
  }

  return { rows, truncated, widths: columnWidths(document, maxCol + 1) }
}

function zipPath(base: string, target: string): string {
  if (target.startsWith('/')) {
    return target.replace(/^\/+/, '')
  }

  const parts = base.split('/').slice(0, -1)

  for (const segment of target.split('/')) {
    if (segment === '..') {
      parts.pop()
    } else if (segment && segment !== '.') {
      parts.push(segment)
    }
  }

  return parts.join('/')
}

function entry(files: Record<string, Uint8Array>, path: string): Uint8Array | undefined {
  return files[path] ?? files[`/${path}`]
}

export function parseXlsx(bytes: Uint8Array): SheetPreview[] {
  let files: Record<string, Uint8Array>

  try {
    files = unzipSync(bytes)
  } catch {
    throw new Error('Could not read this spreadsheet')
  }

  let inflated = 0

  for (const part of Object.values(files)) {
    inflated += part.length

    if (inflated > MAX_UNZIPPED_BYTES) {
      throw new Error('Spreadsheet is too large to preview')
    }
  }

  const workbook = entry(files, 'xl/workbook.xml')

  if (!workbook) {
    throw new Error('Could not read this spreadsheet')
  }

  const document = parseXml(textOf(workbook))
  const strings = sharedStrings(entry(files, 'xl/sharedStrings.xml'))
  const formats = styleFormats(entry(files, 'xl/styles.xml'))
  const rels = parseXml(textOf(entry(files, 'xl/_rels/workbook.xml.rels') || new TextEncoder().encode('<Relationships/>')))
  const targets = new Map<string, string>()

  for (const relationship of descendants(rels, 'Relationship')) {
    const id = relationship.getAttribute('Id')
    const target = relationship.getAttribute('Target')

    if (id && target) {
      targets.set(id, zipPath('xl/workbook.xml', target))
    }
  }

  const sheets = descendants(document, 'sheet').map((sheet, index) => {
    const name = attribute(sheet, 'name') || `Sheet ${index + 1}`
    const id = attribute(sheet, 'id')
    const path = (id && targets.get(id)) || `xl/worksheets/sheet${index + 1}.xml`
    const body = entry(files, path)

    if (!body) {
      return { name, rows: [], truncated: false }
    }

    return { name, ...sheetGrid(body, strings, formats) }
  })

  if (!sheets.length) {
    throw new Error('Could not read this spreadsheet')
  }

  return sheets
}
