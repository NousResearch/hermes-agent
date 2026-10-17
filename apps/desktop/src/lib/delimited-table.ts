// CSV/TSV parsing for the file preview's table view. Ported from
// paperclipai/paperclip#15445 (ui/src/lib/csv-preview.ts), widened to TSV.

export type TableDelimiter = ',' | '\t'

const DELIMITER_BY_EXT: Record<string, TableDelimiter> = { csv: ',', tsv: '\t' }

/** `,` / tab for a `.csv` / `.tsv` path (query/hash ignored), else null. */
export function tableDelimiterForPath(path: string | undefined): TableDelimiter | null {
  const clean = (path || '').split(/[?#]/, 1)[0] || ''
  const ext = /\.([^./\\]+)$/.exec(clean)?.[1]?.toLowerCase() ?? ''

  return DELIMITER_BY_EXT[ext] ?? null
}

export class UnterminatedQuoteError extends Error {}

/** RFC 4180 records: quoted delimiters/newlines, `""` escapes, BOM, CRLF. A
 *  quote opens a quoted field only at the field's start; elsewhere it is literal.
 *  Throws UnterminatedQuoteError when a quoted field never closes. */
export function parseDelimited(text: string, delimiter: TableDelimiter = ','): string[][] {
  const source = text.replace(/^\uFEFF/, '')

  if (!source) {
    return []
  }

  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  let fieldStart = true

  for (let i = 0; i < source.length; i++) {
    const char = source[i]

    if (quoted) {
      if (char !== '"') {
        field += char
      } else if (source[i + 1] === '"') {
        field += '"'
        i++
      } else {
        quoted = false
      }

      continue
    }

    if (char === '"' && fieldStart) {
      quoted = true
      fieldStart = false
    } else if (char === delimiter) {
      row.push(field)
      field = ''
      fieldStart = true
    } else if (char === '\n' || char === '\r') {
      row.push(field)
      rows.push(row)
      row = []
      field = ''
      fieldStart = true

      if (char === '\r' && source[i + 1] === '\n') {
        i++
      }
    } else {
      field += char
      fieldStart = false
    }
  }

  if (quoted) {
    throw new UnterminatedQuoteError('unterminated quoted field')
  }

  // A trailing newline ends the last record; anything after it is one more.
  if (!fieldStart || row.length > 0 || !/[\r\n]$/.test(source)) {
    row.push(field)
    rows.push(row)
  }

  return rows
}
