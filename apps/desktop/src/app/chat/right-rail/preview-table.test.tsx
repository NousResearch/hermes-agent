import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { parseDelimited, UnterminatedQuoteError } from '@/lib/delimited-table'
import { $connection } from '@/store/session'

import { LocalFilePreview } from './preview-file'

vi.mock('@/components/chat/shiki-highlighter', () => ({
  LazyShiki: ({ code }: { code: string }) => <pre data-testid="source">{code}</pre>
}))

afterEach(() => {
  cleanup()
  $connection.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

it('parses RFC 4180 records: quoted delimiters, escaped quotes, multiline cells, BOM, CRLF, TSV', () => {
  expect(parseDelimited('\uFEFFname,note\r\n"Smith, J","said ""hi""\nthen left"\r\nx,\n')).toEqual([
    ['name', 'note'],
    ['Smith, J', 'said "hi"\nthen left'],
    ['x', '']
  ])
  // A quote mid-field is literal, and TSV keeps commas inside a cell.
  expect(parseDelimited('a\t5" pipe, steel\nb\t"q\tz"', '\t')).toEqual([
    ['a', '5" pipe, steel'],
    ['b', 'q\tz']
  ])
  expect(() => parseDelimited('a,"open\n')).toThrow(UnterminatedQuoteError)
})

it('opens a CSV file as a table by default and keeps Source one click away', async () => {
  const text = 'region,revenue\n"North, East",1200\nSouth,900\n'
  $connection.set({ mode: 'local' } as never)
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
    readFileText: vi.fn(async () => ({ byteSize: text.length, path: '/work/sales.csv', text }))
  }

  await act(async () => {
    render(
      <LocalFilePreview
        reloadKey={0}
        target={{
          kind: 'file',
          label: 'sales.csv',
          path: '/work/sales.csv',
          previewKind: 'text',
          source: '/work/sales.csv',
          url: 'file:///work/sales.csv'
        }}
      />
    )
  })

  const table = await screen.findByRole('table')
  expect([...table.querySelectorAll('thead th')].map(cell => cell.textContent)).toEqual(['#', 'region', 'revenue'])
  expect(table.querySelector('tbody td')?.textContent).toBe('North, East')
  expect(screen.getByRole('status').textContent).toContain('2 rows')

  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'SOURCE' }))
  })

  expect(screen.queryByRole('table')).toBeNull()
  expect(screen.getByTestId('source').textContent).toContain('"North, East",1200')
})
