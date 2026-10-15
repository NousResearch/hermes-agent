import type { SheetPreview } from './office-xlsx'

const MAX_ROWS = 200
const MAX_COLS = 40

function delimiter(text: string): string {
  const line = text.split(/\r?\n/, 1)[0] ?? ''

  const counts = [
    [',', (line.match(/,/g) || []).length],
    [';', (line.match(/;/g) || []).length],
    ['\t', (line.match(/\t/g) || []).length]
  ] as const

  const winner = counts.reduce((best, next) => (next[1] > best[1] ? next : best))

  return winner[1] > 0 ? winner[0] : ','
}

export function parseCsv(bytes: Uint8Array, name = 'Sheet'): SheetPreview[] {
  const text = new TextDecoder().decode(bytes).replace(/^\uFEFF/, '')
  const split = delimiter(text)
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  let truncated = false

  const pushRow = () => {
    if (rows.length >= MAX_ROWS) {
      truncated = true

      return
    }

    rows.push(row.slice(0, MAX_COLS))

    if (row.length > MAX_COLS) {
      truncated = true
    }

    row = []
  }

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]

    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') {
          cell += '"'
          index += 1
        } else {
          quoted = false
        }
      } else {
        cell += char
      }

      continue
    }

    if (char === '"') {
      quoted = true
    } else if (char === split) {
      row.push(cell)
      cell = ''
    } else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[index + 1] === '\n') {
        index += 1
      }

      row.push(cell)
      cell = ''
      pushRow()
    } else {
      cell += char
    }
  }

  if (cell.length || row.length) {
    row.push(cell)
    pushRow()
  }

  return [{ name, rows, truncated }]
}
