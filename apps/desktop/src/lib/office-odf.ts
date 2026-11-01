import { unzipSync } from 'fflate'

import { officePageDocument } from './office-document'
import type { SheetPreview } from './office-xlsx'

const MAX_ROWS = 200
const MAX_COLS = 40
const MAX_UNZIPPED_BYTES = 32 * 1024 * 1024

function textOf(bytes: Uint8Array | undefined): string {
  return bytes ? new TextDecoder().decode(bytes) : ''
}

function descendants(root: ParentNode, name: string): Element[] {
  return Array.from(root.querySelectorAll('*')).filter(element => element.localName === name)
}

function childrenNamed(element: Element, name: string): Element[] {
  return Array.from(element.children).filter(child => child.localName === name)
}

function attribute(element: Element, name: string): string | null {
  for (const attr of element.attributes) {
    if (attr.localName === name) {
      return attr.value
    }
  }

  return null
}

function escapeHtml(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function unzip(bytes: Uint8Array, label: string): Record<string, Uint8Array> {
  let files: Record<string, Uint8Array>

  try {
    files = unzipSync(bytes)
  } catch {
    throw new Error(label)
  }

  let inflated = 0

  for (const part of Object.values(files)) {
    inflated += part.length

    if (inflated > MAX_UNZIPPED_BYTES) {
      throw new Error(label)
    }
  }

  return files
}

function contentDocument(files: Record<string, Uint8Array>, label: string): Document {
  const xml = textOf(files['content.xml'] ?? files['/content.xml'])
  const document = new DOMParser().parseFromString(xml, 'application/xml')

  if (!xml.trim() || document.querySelector('parsererror')) {
    throw new Error(label)
  }

  return document
}

function blocks(element: Element): string {
  let html = ''

  for (const child of Array.from(element.children)) {
    const name = child.localName

    if (name === 'h') {
      const level = Math.min(3, Math.max(1, Number(attribute(child, 'outline-level') || 1)))

      html += `<h${level}>${escapeHtml(child.textContent ?? '')}</h${level}>`
    } else if (name === 'p') {
      const text = child.textContent ?? ''

      if (text.trim()) {
        html += `<p>${escapeHtml(text)}</p>`
      }
    } else if (name === 'list') {
      html += '<ul>'

      for (const item of childrenNamed(child, 'list-item')) {
        html += `<li>${escapeHtml(item.textContent ?? '')}</li>`
      }

      html += '</ul>'
    } else if (name === 'table') {
      html += '<table>'

      for (const row of childrenNamed(child, 'table-row')) {
        html += '<tr>'

        for (const cell of childrenNamed(row, 'table-cell')) {
          html += `<td>${escapeHtml(cell.textContent ?? '')}</td>`
        }

        html += '</tr>'
      }

      html += '</table>'
    } else if (name === 'section' || name === 'text') {
      html += blocks(child)
    }
  }

  return html
}

export function odtPreviewDocument(bytes: Uint8Array): string {
  const files = unzip(bytes, 'Could not read this document')
  const document = contentDocument(files, 'Could not read this document')
  const text = descendants(document, 'text')[0]

  if (!text) {
    throw new Error('Could not read this document')
  }

  return officePageDocument(blocks(text))
}

function cellText(cell: Element): string {
  const displayed = descendants(cell, 'p')
    .map(paragraph => paragraph.textContent ?? '')
    .join('\n')
    .trim()

  if (displayed) {
    return displayed
  }

  return attribute(cell, 'value') || attribute(cell, 'boolean-value') || ''
}

function repeatCount(element: Element, name: string): number {
  const repeat = Number(attribute(element, name) || 1)

  return Number.isFinite(repeat) && repeat > 0 ? repeat : 1
}

export function parseOds(bytes: Uint8Array): SheetPreview[] {
  const files = unzip(bytes, 'Could not read this spreadsheet')
  const document = contentDocument(files, 'Could not read this spreadsheet')
  const tables = descendants(document, 'table').filter(table => table.parentElement?.localName === 'spreadsheet')

  const sheets = tables.map((table, index) => {
    const rows: string[][] = []
    let truncated = false

    for (const tableRow of childrenNamed(table, 'table-row')) {
      const rowRepeat = Math.min(repeatCount(tableRow, 'number-rows-repeated'), MAX_ROWS - rows.length)
      const row: string[] = []

      for (const cell of Array.from(tableRow.children)) {
        if (cell.localName !== 'table-cell' && cell.localName !== 'covered-table-cell') {
          continue
        }

        const value = cell.localName === 'covered-table-cell' ? '' : cellText(cell)
        const requested = repeatCount(cell, 'number-columns-repeated')
        const colRepeat = value ? Math.min(requested, MAX_COLS - row.length) : requested === 1 && row.length < MAX_COLS ? 1 : 0

        if (repeatCount(cell, 'number-columns-repeated') > colRepeat) {
          truncated = true
        }

        for (let copy = 0; copy < colRepeat; copy += 1) {
          row.push(value)
        }

        if (row.length >= MAX_COLS) {
          break
        }
      }

      for (let copy = 0; copy < rowRepeat; copy += 1) {
        rows.push(row)
      }

      if (repeatCount(tableRow, 'number-rows-repeated') > rowRepeat) {
        truncated = true
      }

      if (rows.length >= MAX_ROWS) {
        truncated = true

        break
      }
    }

    return {
      name: attribute(table, 'name') || `Sheet ${index + 1}`,
      rows,
      truncated
    }
  })

  if (!sheets.length) {
    throw new Error('Could not read this spreadsheet')
  }

  return sheets
}
