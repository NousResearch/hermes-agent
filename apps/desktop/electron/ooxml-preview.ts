import fs from 'node:fs'
import zlib from 'node:zlib'

// .docx / .pptx / .xlsx are OOXML: plain ZIP containers around XML parts. This
// extracts read-only plain text from the parts that matter for a preview —
// no rendering, no formulas, no third-party ZIP/XML dependency (see #81159).
export const OOXML_PREVIEW_EXTENSIONS = new Set(['.docx', '.pptx', '.xlsx'])

// A single part's decompressed XML is virtually always well under this even
// for large documents (embedded media lives in separate, untouched ZIP
// entries) — capping it turns a hostile "declared size lies" ZIP entry into a
// thrown error instead of an unbounded inflate.
const MAX_PART_BYTES = 8 * 1024 * 1024
const MAX_SLIDES = 300
const MAX_SHEETS = 50
const MAX_ROWS_PER_SHEET = 2000

// Extraction reads the whole archive into memory before any other cap
// applies (the ZIP central directory can be anywhere in the file, so there's
// no way to stream just the parts we need without a full ZIP implementation).
// Bound that up front by file size so a large archive falls back to the
// existing bounded (512 KiB) binary-guard read instead of being buffered
// whole in the Electron main process.
export const MAX_OOXML_SOURCE_BYTES = 20 * 1024 * 1024

const EOCD_SIGNATURE = 0x06054b50
const CENTRAL_DIR_SIGNATURE = 0x02014b50
const LOCAL_HEADER_SIGNATURE = 0x04034b50
const EOCD_RECORD_MIN_BYTES = 22
const EOCD_MAX_COMMENT_BYTES = 0xffff

interface ZipEntry {
  compressedSize: number
  compressionMethod: number
  localHeaderOffset: number
  name: string
}

const XML_NAMED_ENTITIES: Record<string, string> = { amp: '&', apos: "'", gt: '>', lt: '<', quot: '"' }

function decodeXmlEntities(value: string): string {
  return value.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (match, entity: string) => {
    if (entity[0] !== '#') {
      return XML_NAMED_ENTITIES[entity] ?? match
    }

    const isHex = entity[1] === 'x' || entity[1] === 'X'
    const codePoint = parseInt(entity.slice(isHex ? 2 : 1), isHex ? 16 : 10)

    return Number.isFinite(codePoint) ? String.fromCodePoint(codePoint) : match
  })
}

function findEndOfCentralDirectory(buffer: Buffer): number {
  const searchStart = Math.max(0, buffer.length - EOCD_RECORD_MIN_BYTES - EOCD_MAX_COMMENT_BYTES)

  for (let offset = buffer.length - EOCD_RECORD_MIN_BYTES; offset >= searchStart; offset -= 1) {
    if (buffer.readUInt32LE(offset) === EOCD_SIGNATURE) {
      return offset
    }
  }

  throw new Error('Not a valid ZIP archive (no end-of-central-directory record found)')
}

function listZipEntries(buffer: Buffer): ZipEntry[] {
  const eocdOffset = findEndOfCentralDirectory(buffer)
  const entryCount = buffer.readUInt16LE(eocdOffset + 10)
  const entries: ZipEntry[] = []
  let offset = buffer.readUInt32LE(eocdOffset + 16)

  for (let i = 0; i < entryCount; i += 1) {
    if (offset + 46 > buffer.length || buffer.readUInt32LE(offset) !== CENTRAL_DIR_SIGNATURE) {
      break
    }

    const compressionMethod = buffer.readUInt16LE(offset + 10)
    const compressedSize = buffer.readUInt32LE(offset + 20)
    const nameLength = buffer.readUInt16LE(offset + 28)
    const extraLength = buffer.readUInt16LE(offset + 30)
    const commentLength = buffer.readUInt16LE(offset + 32)
    const localHeaderOffset = buffer.readUInt32LE(offset + 42)
    const name = buffer.toString('utf8', offset + 46, offset + 46 + nameLength)

    entries.push({ compressedSize, compressionMethod, localHeaderOffset, name })
    offset += 46 + nameLength + extraLength + commentLength
  }

  return entries
}

function readZipEntryText(buffer: Buffer, entry: ZipEntry): string {
  const offset = entry.localHeaderOffset

  if (offset + 30 > buffer.length || buffer.readUInt32LE(offset) !== LOCAL_HEADER_SIGNATURE) {
    throw new Error(`Corrupt ZIP local header for "${entry.name}"`)
  }

  const nameLength = buffer.readUInt16LE(offset + 26)
  const extraLength = buffer.readUInt16LE(offset + 28)
  const dataStart = offset + 30 + nameLength + extraLength
  const compressed = buffer.subarray(dataStart, dataStart + entry.compressedSize)

  if (entry.compressionMethod === 0) {
    if (compressed.length > MAX_PART_BYTES) {
      throw new Error(`ZIP part "${entry.name}" exceeds the preview extraction cap`)
    }

    return compressed.toString('utf8')
  }

  if (entry.compressionMethod === 8) {
    // maxOutputLength bounds the inflated result itself, regardless of what
    // the (attacker-controllable) declared uncompressed size claims.
    return zlib.inflateRawSync(compressed, { maxOutputLength: MAX_PART_BYTES }).toString('utf8')
  }

  throw new Error(`Unsupported ZIP compression method ${entry.compressionMethod} for "${entry.name}"`)
}

function findEntry(entries: ZipEntry[], name: string): ZipEntry | undefined {
  return entries.find(entry => entry.name === name)
}

function partNumber(name: string, pattern: RegExp): number {
  const match = pattern.exec(name)

  return match ? Number(match[1]) : 0
}

function extractDocxText(buffer: Buffer, entries: ZipEntry[]): string | null {
  const entry = findEntry(entries, 'word/document.xml')

  if (!entry) {
    return null
  }

  const xml = readZipEntryText(buffer, entry)

  const paragraphs = xml.split('</w:p>').map(paragraph => {
    const runs = paragraph.match(/<w:t[^>]*>[\s\S]*?<\/w:t>/g) || []

    return runs.map(run => decodeXmlEntities(run.replace(/<w:t[^>]*>/, '').replace(/<\/w:t>$/, ''))).join('')
  })

  return paragraphs
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function extractPptxText(buffer: Buffer, entries: ZipEntry[]): string | null {
  const slidePattern = /^ppt\/slides\/slide(\d+)\.xml$/

  const slideEntries = entries
    .filter(entry => slidePattern.test(entry.name))
    .sort((a, b) => partNumber(a.name, slidePattern) - partNumber(b.name, slidePattern))
    .slice(0, MAX_SLIDES)

  if (!slideEntries.length) {
    return null
  }

  const slides = slideEntries.map(entry => {
    const xml = readZipEntryText(buffer, entry)
    const runs = xml.match(/<a:t>[\s\S]*?<\/a:t>/g) || []

    return runs.map(run => decodeXmlEntities(run.slice(5, -6))).join(' ')
  })

  return slides
    .map((text, index) => `--- Slide ${index + 1} ---\n${text}`.trimEnd())
    .join('\n\n')
    .trim()
}

function extractSharedStrings(buffer: Buffer, entries: ZipEntry[]): string[] {
  const entry = findEntry(entries, 'xl/sharedStrings.xml')

  if (!entry) {
    return []
  }

  const xml = readZipEntryText(buffer, entry)
  const items = xml.match(/<si>[\s\S]*?<\/si>/g) || []

  return items.map(item => {
    const runs = item.match(/<t[^>]*>[\s\S]*?<\/t>/g) || []

    return decodeXmlEntities(runs.map(run => run.replace(/<t[^>]*>/, '').replace(/<\/t>$/, '')).join(''))
  })
}

function columnIndexFromCellRef(ref: string): number {
  const letters = /^([A-Z]+)/.exec(ref)?.[1] || ''
  let index = 0

  for (const char of letters) {
    index = index * 26 + (char.charCodeAt(0) - 64)
  }

  return index > 0 ? index - 1 : 0
}

function extractSheetRows(xml: string, sharedStrings: string[]): string[][] {
  const rowMatches = xml.match(/<row\b[^>]*>[\s\S]*?<\/row>/g) || []

  return rowMatches.slice(0, MAX_ROWS_PER_SHEET).map(rowXml => {
    const cellMatches = rowXml.match(/<c\b[^>]*\/>|<c\b[^>]*>[\s\S]*?<\/c>/g) || []
    const row: string[] = []

    cellMatches.forEach((cellXml, cellIndex) => {
      const ref = /\br="([A-Z]+\d+)"/.exec(cellXml)?.[1]
      const column = ref ? columnIndexFromCellRef(ref) : cellIndex
      const type = /\bt="([^"]+)"/.exec(cellXml)?.[1]

      let value = ''

      if (type === 's') {
        const sharedIndex = /<v>([\s\S]*?)<\/v>/.exec(cellXml)?.[1]
        value = sharedIndex !== undefined ? (sharedStrings[Number(sharedIndex)] ?? '') : ''
      } else if (type === 'inlineStr') {
        value = decodeXmlEntities(/<t[^>]*>([\s\S]*?)<\/t>/.exec(cellXml)?.[1] ?? '')
      } else {
        value = /<v>([\s\S]*?)<\/v>/.exec(cellXml)?.[1] ?? ''
      }

      while (row.length < column) {
        row.push('')
      }

      row[column] = value
    })

    return row
  })
}

function extractXlsxText(buffer: Buffer, entries: ZipEntry[]): string | null {
  const sheetPattern = /^xl\/worksheets\/sheet(\d+)\.xml$/

  const sheetEntries = entries
    .filter(entry => sheetPattern.test(entry.name))
    .sort((a, b) => partNumber(a.name, sheetPattern) - partNumber(b.name, sheetPattern))
    .slice(0, MAX_SHEETS)

  if (!sheetEntries.length) {
    return null
  }

  const sharedStrings = extractSharedStrings(buffer, entries)

  const sheets = sheetEntries.map((entry, index) => {
    const xml = readZipEntryText(buffer, entry)
    const rows = extractSheetRows(xml, sharedStrings)
    const body = rows.map(row => row.join('\t')).join('\n')

    return `--- Sheet ${index + 1} ---\n${body}`.trimEnd()
  })

  return sheets.join('\n\n').trim()
}

const OOXML_EXTRACTORS: Record<string, (buffer: Buffer, entries: ZipEntry[]) => string | null> = {
  '.docx': extractDocxText,
  '.pptx': extractPptxText,
  '.xlsx': extractXlsxText
}

/** Pure, buffer-in/string-out — the part a unit test drives directly. */
export function extractOoxmlPreviewTextFromBuffer(buffer: Buffer, ext: string): string | null {
  const extractor = OOXML_EXTRACTORS[ext.toLowerCase()]

  if (!extractor) {
    return null
  }

  const entries = listZipEntries(buffer)
  const text = extractor(buffer, entries)

  return text && text.trim() ? text : null
}

/**
 * Best-effort OOXML text extraction for the desktop file preview pane.
 * Returns null on any failure (corrupt archive, legacy binary format saved
 * with a modern extension, encrypted document, unsupported layout) so the
 * caller falls back to the existing binary-file preview guard.
 */
export async function extractOoxmlPreviewText(
  filePath: string,
  ext: string,
  sourceByteSize?: number
): Promise<string | null> {
  try {
    const size = sourceByteSize ?? (await fs.promises.stat(filePath)).size

    if (size > MAX_OOXML_SOURCE_BYTES) {
      return null
    }

    const buffer = await fs.promises.readFile(filePath)

    return extractOoxmlPreviewTextFromBuffer(buffer, ext)
  } catch {
    return null
  }
}
