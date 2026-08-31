import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { OfficePreviewView } from './preview-office'

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
              rows: [
                ['Name', '42'],
                ['Ada', 'TRUE']
              ]
            },
            { name: 'Notes', rows: [['hello']] }
          ]
        }}
        truncatedLabel="truncated"
      />
    )

    expect(screen.getByText('Name')).toBeTruthy()
    expect(screen.getByText('42')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Notes' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Notes' }))
    expect(screen.getByText('hello')).toBeTruthy()
    expect(screen.queryByText('Ada')).toBeNull()
  })

  it('renders document HTML without executing injected markup', () => {
    const { container } = render(
      <OfficePreviewView
        preview={{ kind: 'document', html: '<p>Hello <strong>Ada</strong></p>' }}
        truncatedLabel="truncated"
      />
    )

    expect(container.querySelector('strong')?.textContent).toBe('Ada')
    expect(screen.getByText('Ada')).toBeTruthy()
  })

  it('renders slides in order', () => {
    render(
      <OfficePreviewView
        preview={{
          kind: 'slides',
          slides: [{ html: '<p>First</p>' }, { html: '<p>Second</p>' }]
        }}
        truncatedLabel="truncated"
      />
    )

    expect(screen.getByText('Slide 1')).toBeTruthy()
    expect(screen.getByText('First')).toBeTruthy()
    expect(screen.getByText('Second')).toBeTruthy()
  })

  it('shows a truncation banner when the parser capped the document', () => {
    render(
      <OfficePreviewView
        preview={{ kind: 'spreadsheet', sheets: [{ name: 'A', rows: [['x']] }], truncated: true }}
        truncatedLabel="Showing a preview of the first sheets, rows, or slides."
      />
    )

    expect(screen.getByText('Showing a preview of the first sheets, rows, or slides.')).toBeTruthy()
  })
})
