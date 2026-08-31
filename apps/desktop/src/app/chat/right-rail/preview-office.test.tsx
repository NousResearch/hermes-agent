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
        slideLabel={index => `Slide ${index}`}
        truncatedLabel="truncated"
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
        slideLabel={index => `Slide ${index}`}
        truncatedLabel="truncated"
      />
    )

    expect(container.querySelector('h1')?.textContent).toBe('Title')
    expect(container.querySelector('strong')?.textContent).toBe('Ada')
    expect(screen.getByText('Ada')).toBeTruthy()
    expect(container.querySelector('[data-office-html]')).toBeNull()
  })

  it('keeps injected markup as text nodes', () => {
    const { container } = render(
      <OfficePreviewView
        preview={{
          blocks: [{ runs: [{ text: '<img src=x onerror=alert(1)>' }], type: 'paragraph' }],
          kind: 'document'
        }}
        slideLabel={index => `Slide ${index}`}
        truncatedLabel="truncated"
      />
    )

    expect(container.querySelector('img')).toBeNull()
    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy()
  })

  it('renders slides in order', () => {
    render(
      <OfficePreviewView
        preview={{
          kind: 'slides',
          slides: [{ lines: ['First'] }, { lines: ['Second'] }]
        }}
        slideLabel={index => `Slide ${index}`}
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
        slideLabel={index => `Slide ${index}`}
        truncatedLabel="Showing a preview of the first sheets, rows, or slides."
      />
    )

    expect(screen.getByText('Showing a preview of the first sheets, rows, or slides.')).toBeTruthy()
  })
})
