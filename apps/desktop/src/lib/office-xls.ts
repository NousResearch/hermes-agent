import CFB from 'cfb'

import type { SheetPreview } from './office-xlsx'

const CONTINUE = 0x003c
const SST = 0x00fc
const BOUNDSHEET = 0x0085
const BOF = 0x0809
const LABELSST = 0x00fd
const LABEL = 0x0204
const NUMBER = 0x0203
const RK = 0x027e
const MULRK = 0x00bd
const BOOLERR = 0x0205
const WORKSHEET = 0x0010
const MAX_ROWS = 200
const MAX_COLS = 40

interface BiffRecord {
  code: number
  chunks: Uint8Array[]
}

function u16(bytes: Uint8Array, offset: number): number {
  return bytes[offset] | (bytes[offset + 1] << 8)
}

function records(buf: Uint8Array): BiffRecord[] {
  const out: BiffRecord[] = []
  let offset = 0

  while (offset + 4 <= buf.length) {
    const code = u16(buf, offset)
    const length = u16(buf, offset + 2)
    const end = Math.min(buf.length, offset + 4 + length)
    const data = buf.subarray(offset + 4, end)
    offset = end

    if (code === CONTINUE && out.length) {
      out[out.length - 1].chunks.push(data)
    } else {
      out.push({ code, chunks: [data] })
    }
  }

  return out
}

class Cursor {
  private index = 0
  private pos = 0

  constructor(
    private readonly chunks: Uint8Array[],
    start: number
  ) {
    this.pos = start
  }

  private ensure() {
    while (this.index < this.chunks.length && this.pos >= this.chunks[this.index].length) {
      this.index += 1
      this.pos = 0
    }
  }

  exhausted(): boolean {
    this.ensure()

    return this.index >= this.chunks.length
  }

  remainingInChunk(): number {
    this.ensure()

    if (this.index >= this.chunks.length) {
      return 0
    }

    return this.chunks[this.index].length - this.pos
  }

  nextChunk() {
    this.index += 1
    this.pos = 0
    this.ensure()
  }

  read(count: number): Uint8Array {
    const out = new Uint8Array(count)
    let filled = 0

    while (filled < count) {
      this.ensure()

      if (this.index >= this.chunks.length) {
        throw new Error('Could not read this spreadsheet')
      }

      const chunk = this.chunks[this.index]
      const take = Math.min(count - filled, chunk.length - this.pos)
      out.set(chunk.subarray(this.pos, this.pos + take), filled)
      this.pos += take
      filled += take
    }

    return out
  }

  readU8(): number {
    return this.read(1)[0]
  }

  readU16(): number {
    const bytes = this.read(2)

    return bytes[0] | (bytes[1] << 8)
  }

  readU32(): number {
    const bytes = this.read(4)

    return (bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)) >>> 0
  }

  readF64(): number {
    return new DataView(this.read(8).buffer).getFloat64(0, true)
  }
}

function decodeChars(bytes: Uint8Array, wide: boolean): string {
  let text = ''

  if (!wide) {
    for (const byte of bytes) {
      text += String.fromCharCode(byte)
    }

    return text
  }

  for (let index = 0; index + 1 < bytes.length; index += 2) {
    text += String.fromCharCode(bytes[index] | (bytes[index + 1] << 8))
  }

  return text
}

function readUnicode(cursor: Cursor, charactersByte: boolean): string {
  const count = charactersByte ? cursor.readU8() : cursor.readU16()
  let wide = (cursor.readU8() & 1) !== 0
  let left = count
  let text = ''

  while (left > 0) {
    const available = cursor.remainingInChunk()
    const bytesPer = wide ? 2 : 1
    const take = Math.min(left, Math.floor(available / bytesPer))

    if (take === 0) {
      cursor.nextChunk()
      wide = (cursor.readU8() & 1) !== 0

      continue
    }

    text += decodeChars(cursor.read(take * bytesPer), wide)
    left -= take
  }

  return text
}

function sharedStrings(record: BiffRecord): string[] {
  const cursor = new Cursor(record.chunks, 8)
  const unique = new DataView(record.chunks[0].buffer, record.chunks[0].byteOffset + 4, 4).getUint32(0, true)
  const strings: string[] = []

  for (let index = 0; index < unique && !cursor.exhausted(); index += 1) {
    const count = cursor.readU16()
    const flags = cursor.readU8()
    const rich = (flags & 0x08) !== 0
    const phonetic = (flags & 0x04) !== 0
    const runs = rich ? cursor.readU16() : 0
    const phoneticBytes = phonetic ? cursor.readU32() : 0
    let wide = (flags & 1) !== 0
    let left = count
    let text = ''

    while (left > 0) {
      const available = cursor.remainingInChunk()
      const bytesPer = wide ? 2 : 1
      const take = Math.min(left, Math.floor(available / bytesPer))

      if (take === 0) {
        cursor.nextChunk()
        wide = (cursor.readU8() & 1) !== 0

        continue
      }

      text += decodeChars(cursor.read(take * bytesPer), wide)
      left -= take
    }

    if (runs) {
      cursor.read(runs * 4)
    }

    if (phoneticBytes) {
      cursor.read(phoneticBytes)
    }

    strings.push(text)
  }

  return strings
}

function decodeRk(value: number): number {
  const divided = (value & 1) !== 0
  let number: number

  if ((value & 2) !== 0) {
    number = value >> 2
  } else {
    const buffer = new ArrayBuffer(8)
    new DataView(buffer).setUint32(4, value & ~3, true)
    number = new DataView(buffer).getFloat64(0, true)
  }

  return divided ? number / 100 : number
}

function put(
  grid: Map<number, Map<number, string>>,
  row: number,
  col: number,
  value: string
): boolean {
  if (row >= MAX_ROWS || col >= MAX_COLS || row < 0 || col < 0) {
    return true
  }

  const cells = grid.get(row) ?? new Map<number, string>()
  cells.set(col, value)
  grid.set(row, cells)

  return false
}

function materialize(grid: Map<number, Map<number, string>>, truncated: boolean): Pick<SheetPreview, 'rows' | 'truncated'> {
  let maxRow = -1
  let maxCol = -1

  for (const [row, cells] of grid) {
    maxRow = Math.max(maxRow, row)

    for (const col of cells.keys()) {
      maxCol = Math.max(maxCol, col)
    }
  }

  const rows: string[][] = []

  for (let row = 0; row <= maxRow; row += 1) {
    const source = grid.get(row)
    rows.push(Array.from({ length: maxCol + 1 }, (_, col) => source?.get(col) ?? ''))
  }

  return { rows, truncated }
}

export function parseBiff8(stream: Uint8Array): SheetPreview[] {
  const parsed = records(stream)
  const names: string[] = []
  let strings: string[] = []
  const sheets: SheetPreview[] = []
  let grid = new Map<number, Map<number, string>>()
  let truncated = false
  let inSheet = false

  const flush = () => {
    if (!inSheet) {
      return
    }

    sheets.push({
      name: names[sheets.length] || `Sheet ${sheets.length + 1}`,
      ...materialize(grid, truncated)
    })
    grid = new Map()
    truncated = false
    inSheet = false
  }

  for (const record of parsed) {
    const data = record.chunks[0] ?? new Uint8Array()

    if (record.code === BOF) {
      const kind = data.length >= 4 ? u16(data, 2) : 0

      if (u16(data, 0) !== 0x0600) {
        throw new Error('This Excel file is too old to preview')
      }

      flush()
      inSheet = kind === WORKSHEET

      continue
    }

    if (record.code === BOUNDSHEET && data.length >= 8) {
      names.push(readUnicode(new Cursor([data.subarray(6)], 0), true))

      continue
    }

    if (record.code === SST) {
      strings = sharedStrings(record)

      continue
    }

    if (!inSheet || data.length < 6) {
      continue
    }

    const row = u16(data, 0)
    const col = u16(data, 2)

    if (record.code === LABELSST && data.length >= 10) {
      const index = new DataView(data.buffer, data.byteOffset + 6, 4).getUint32(0, true)
      truncated = put(grid, row, col, strings[index] ?? '') || truncated
    } else if (record.code === LABEL) {
      truncated = put(grid, row, col, readUnicode(new Cursor(record.chunks, 6), false)) || truncated
    } else if (record.code === NUMBER && data.length >= 14) {
      truncated = put(grid, row, col, String(new DataView(data.buffer, data.byteOffset + 6, 8).getFloat64(0, true))) || truncated
    } else if (record.code === RK && data.length >= 10) {
      truncated =
        put(grid, row, col, String(decodeRk(new DataView(data.buffer, data.byteOffset + 6, 4).getUint32(0, true)))) ||
        truncated
    } else if (record.code === MULRK && data.length >= 6) {
      let offset = 4

      while (offset + 6 <= data.length - 2) {
        const value = new DataView(data.buffer, data.byteOffset + offset + 2, 4).getUint32(0, true)
        truncated = put(grid, row, col + (offset - 4) / 6, String(decodeRk(value))) || truncated
        offset += 6
      }
    } else if (record.code === BOOLERR && data.length >= 8 && data[7] === 0) {
      truncated = put(grid, row, col, data[6] ? 'TRUE' : 'FALSE') || truncated
    }
  }

  flush()

  if (!sheets.length) {
    throw new Error('Could not read this spreadsheet')
  }

  return sheets
}

function streamBytes(content: Uint8Array | number[] | undefined): Uint8Array | null {
  if (!content) {
    return null
  }

  return content instanceof Uint8Array ? content : Uint8Array.from(content)
}

export function parseXls(bytes: Uint8Array): SheetPreview[] {
  let container: ReturnType<typeof CFB.read>

  try {
    container = CFB.read(bytes, { type: 'array' })
  } catch {
    throw new Error('Could not read this spreadsheet')
  }

  const entry =
    CFB.find(container, '/Workbook') ||
    CFB.find(container, 'Workbook') ||
    CFB.find(container, '/Book') ||
    CFB.find(container, 'Book')

  const stream = streamBytes(entry?.content)

  if (!stream) {
    throw new Error('Could not read this spreadsheet')
  }

  return parseBiff8(stream)
}
