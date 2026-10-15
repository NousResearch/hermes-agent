import { parseCsv } from './office-csv'
import { parseOds } from './office-odf'
import { parseXls } from './office-xls'
import { parseXlsx, type SheetPreview } from './office-xlsx'

export type { SheetPreview }

export function parseSpreadsheet(bytes: Uint8Array, name?: string): SheetPreview[] {
  if (name && /\.ods$/i.test(name)) {
    return parseOds(bytes)
  }

  if (bytes.length >= 2 && bytes[0] === 0x50 && bytes[1] === 0x4b) {
    return parseXlsx(bytes)
  }

  if (bytes.length >= 4 && bytes[0] === 0xd0 && bytes[1] === 0xcf && bytes[2] === 0x11 && bytes[3] === 0xe0) {
    return parseXls(bytes)
  }

  if (name && /\.csv$/i.test(name)) {
    return parseCsv(bytes, name.replace(/\.[^.]+$/, '') || 'Sheet')
  }

  throw new Error('Could not read this spreadsheet')
}

export function dataUrlBytes(dataUrl: string): Uint8Array {
  const comma = dataUrl.indexOf(',')

  if (comma < 0 || !dataUrl.startsWith('data:')) {
    throw new Error('Invalid office preview data')
  }

  const payload = dataUrl.slice(comma + 1)
  let binary: string

  try {
    binary = atob(decodeURIComponent(payload))
  } catch {
    try {
      binary = atob(payload)
    } catch {
      throw new Error('Invalid office preview data')
    }
  }

  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }

  return bytes
}
