import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { SpreadsheetCell } from '@/lib/ooxml-preview'

import { OfficePreviewView } from './preview-office'

function cells(rows: Array<Array<string | SpreadsheetCell>>): SpreadsheetCell[][] {
  return rows.map(row => row.map(cell => (typeof cell === 'string' ? { value: cell } : cell)))
}

const labels = {
  formulaBarLabel: 'Formula',
  truncatedLabel: 'truncated'
}

describe('OfficePreviewView', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders spreadsheet cells and switches sheets', () => {
    render(
      <OfficePreviewView
        preview={{
          kind: 'spreadsheet',
          sheets: [
            {
              name: 'Revenue',
              rows: cells([
                ['Name', '42'],
                ['Ada', 'TRUE']
              ])
            },
            { name: 'Notes', rows: cells([['hello']]) }
          ]
        }}
        {...labels}
      />
    )

    expect(screen.getByText('Name')).toBeTruthy()
    expect(screen.getByText('42')).toBeTruthy()
    expect(screen.getByRole('tab', { name: 'Notes' })).toBeTruthy()
    expect(screen.getByRole('tab', { name: 'Revenue' }).getAttribute('aria-selected')).toBe('true')

    fireEvent.click(screen.getByRole('tab', { name: 'Notes' }))
    expect(screen.getByText('hello')).toBeTruthy()
    expect(screen.queryByText('Ada')).toBeNull()
    expect(screen.getByRole('tab', { name: 'Notes' }).getAttribute('aria-selected')).toBe('true')
  })

  it('shows the stored formula in a read-only bar when a cell is selected', () => {
    render(
      <OfficePreviewView
        preview={{
          kind: 'spreadsheet',
          sheets: [
            {
              name: 'Revenue',
              rows: cells([['Name', { formula: 'B1+1', value: '43' }]])
            }
          ]
        }}
        {...labels}
      />
    )

    fireEvent.click(screen.getByRole('gridcell', { name: '43' }))

    expect((screen.getByLabelText('Formula') as HTMLInputElement).value).toBe('=B1+1')
    expect(screen.getByText('B1')).toBeTruthy()
  })

  it('shows the cell value in the formula bar when there is no formula', () => {
    render(
      <OfficePreviewView
        preview={{
          kind: 'spreadsheet',
          sheets: [{ name: 'Revenue', rows: cells([['Ada']]) }]
        }}
        {...labels}
      />
    )

    fireEvent.click(screen.getByRole('gridcell', { name: 'Ada' }))

    expect((screen.getByLabelText('Formula') as HTMLInputElement).value).toBe('Ada')
    expect(screen.getByText('A1')).toBeTruthy()
  })

  it('paints fill, color, and bold from cell formatting', () => {
    render(
      <OfficePreviewView
        preview={{
          kind: 'spreadsheet',
          sheets: [
            {
              name: 'Revenue',
              rows: cells([[{ bold: true, color: '#FF0000', fill: '#C6EFCE', value: 'Ada' }]])
            }
          ]
        }}
        {...labels}
      />
    )

    const cell = screen.getByRole('gridcell', { name: 'Ada' })

    expect(cell.style.backgroundColor).toBe('rgb(198, 239, 206)')
    expect(cell.style.color).toBe('rgb(255, 0, 0)')
    expect(cell.style.fontWeight).toBe('700')
  })

  it('paints unstyled cells on a white sheet so dark Excel fonts stay readable', () => {
    render(
      <OfficePreviewView
        preview={{
          kind: 'spreadsheet',
          sheets: [{ name: 'Revenue', rows: cells([['Ada']])}]
        }}
        {...labels}
      />
    )

    const cell = screen.getByRole('gridcell', { name: 'Ada' })

    expect(cell.style.backgroundColor).toBe('rgb(255, 255, 255)')
    expect(cell.style.color).toBe('rgb(0, 0, 0)')
    expect(screen.getByTestId('office-sheet-scroll').className).toMatch(/overflow-auto/)
    expect(screen.getByTestId('office-sheet-scroll').style.backgroundColor).toBe('rgb(255, 255, 255)')
  })

  it('keeps the sheet wide enough to scroll past the pane', () => {
    render(
      <OfficePreviewView
        preview={{
          kind: 'spreadsheet',
          sheets: [{ name: 'Revenue', rows: cells([['Ada', 'Bob', 'Cara']])}]
        }}
        {...labels}
      />
    )

    const table = screen.getByRole('grid')

    expect(table.className).toMatch(/w-max/)
    expect(screen.getByRole('gridcell', { name: 'Ada' }).className).toMatch(/min-w-/)
  })

  it('renders spreadsheet cells in a Calibri-like sans, not the Desktop mono stack', () => {
    render(
      <OfficePreviewView
        preview={{
          kind: 'spreadsheet',
          sheets: [{ name: 'Revenue', rows: cells([['Ada']])}]
        }}
        {...labels}
      />
    )

    const table = screen.getByRole('grid')

    expect(table.className).not.toMatch(/font-mono/)
    expect(table.style.fontFamily).toMatch(/Calibri/i)
  })

  it('renders document runs as React text, not HTML', () => {
    const { container } = render(
      <OfficePreviewView
        preview={{
          blocks: [
            { heading: 1, runs: [{ text: 'Title' }], type: 'paragraph' },
            { runs: [{ text: 'Hello ' }, { bold: true, text: 'Ada' }], type: 'paragraph' },
            { rows: [['A', 'B']], type: 'table' }
          ],
          kind: 'document'
        }}
        {...labels}
      />
    )

    expect(container.querySelector('h1')?.textContent).toBe('Title')
    expect(screen.getByText('Ada').style.fontWeight).toBe('700')
    expect(screen.getByText('Ada')).toBeTruthy()
    expect(container.querySelector('[data-office-html]')).toBeNull()
  })

  it('renders the document on a white page with a Calibri-like body font', () => {
    render(
      <OfficePreviewView
        preview={{
          blocks: [{ runs: [{ text: 'Hello' }], type: 'paragraph' }],
          kind: 'document'
        }}
        {...labels}
      />
    )

    const page = screen.getByTestId('office-document-page')

    expect(page.style.backgroundColor).toBe('rgb(255, 255, 255)')
    expect(page.style.color).toBe('rgb(0, 0, 0)')
    expect(page.style.fontFamily).toMatch(/Calibri/i)
    expect(screen.getByTestId('office-document-scroll').className).toMatch(/overflow-auto/)
  })

  it('paints Word run formatting and bullet lists on the page', () => {
    render(
      <OfficePreviewView
        preview={{
          blocks: [
            {
              align: 'center',
              runs: [
                {
                  color: '#0000FF',
                  fontFamily: 'Georgia',
                  fontSize: 14,
                  italic: true,
                  text: 'Hello',
                  underline: true
                }
              ],
              type: 'paragraph'
            },
            { list: 'bullet', runs: [{ text: 'Item' }], type: 'paragraph' }
          ],
          kind: 'document'
        }}
        {...labels}
      />
    )

    const hello = screen.getByText('Hello')

    expect(hello.style.color).toBe('rgb(0, 0, 255)')
    expect(hello.style.fontFamily).toMatch(/Georgia/)
    expect(hello.style.fontSize).toBe('14pt')
    expect(hello.style.fontStyle).toBe('italic')
    expect(hello.style.textDecoration).toMatch(/underline/)
    expect(screen.getByText('Item').closest('li')).toBeTruthy()
  })

  it('keeps CJK font names and paints RTL paragraphs', () => {
    render(
      <OfficePreviewView
        preview={{
          blocks: [
            {
              align: 'right',
              dir: 'rtl',
              runs: [{ fontFamily: '微软雅黑, Arial, Calibri', text: 'مرحبا' }],
              type: 'paragraph'
            }
          ],
          kind: 'document'
        }}
        {...labels}
      />
    )

    const run = screen.getByText('مرحبا')
    const paragraph = run.closest('p')

    expect(run.style.fontFamily).toMatch(/微软雅黑/)
    expect(paragraph?.getAttribute('dir')).toBe('rtl')
    expect(paragraph?.style.direction).toBe('rtl')
    expect(paragraph?.style.textAlign).toBe('right')
  })

  it('keeps injected markup as text nodes', () => {
    const { container } = render(
      <OfficePreviewView
        preview={{
          blocks: [{ runs: [{ text: '<img src=x onerror=alert(1)>' }], type: 'paragraph' }],
          kind: 'document'
        }}
        {...labels}
      />
    )

    expect(container.querySelector('img')).toBeNull()
    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy()
  })

  it('shows a truncation banner when the parser capped the document', () => {
    render(
      <OfficePreviewView
        formulaBarLabel="Formula"
        preview={{ kind: 'spreadsheet', sheets: [{ name: 'A', rows: cells([['x']]) }], truncated: true }}
        truncatedLabel="Showing a preview of the first sheets or rows."
      />
    )

    expect(screen.getByText('Showing a preview of the first sheets or rows.')).toBeTruthy()
  })
})
