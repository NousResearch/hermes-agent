import { describe, expect, it } from 'vitest'

import { decodePreviewTabs } from './preview'

describe('persisted preview migration', () => {
  it('upgrades a pre-PDF remote tab from binary to pdf', () => {
    const source = '/remote/.hermes/desktop-attachments/spec.pdf'

    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: `file:file://${source}`,
          target: {
            binary: true,
            kind: 'file',
            label: 'spec.pdf',
            large: true,
            path: source,
            previewKind: 'binary',
            source,
            url: `file://${source}`
          }
        }
      ])
    )

    expect(restored?.target.previewKind).toBe('pdf')
  })

  it('leaves a persisted non-PDF binary tab unchanged', () => {
    const source = '/work/archive.zip'

    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: `file:file://${source}`,
          target: {
            binary: true,
            kind: 'file',
            label: 'archive.zip',
            path: source,
            previewKind: 'binary',
            source,
            url: `file://${source}`
          }
        }
      ])
    )

    expect(restored?.target.previewKind).toBe('binary')
  })

  it.each(['report.pdf#notes', 'report.pdf?draft'])('treats %s as a literal filesystem path', sourceName => {
    const source = `/work/${sourceName}`

    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: `file:file://${encodeURI(source)}`,
          target: {
            binary: true,
            kind: 'file',
            label: sourceName,
            path: source,
            previewKind: 'binary',
            source,
            url: `file:///work/${encodeURIComponent(sourceName)}`
          }
        }
      ])
    )

    expect(restored?.target.previewKind).toBe('binary')
  })

  it('does not overwrite a non-binary PDF preview kind', () => {
    const source = '/work/spec.pdf'

    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: `file:file://${source}`,
          target: {
            kind: 'file',
            label: 'spec.pdf',
            path: source,
            previewKind: 'text',
            source,
            url: `file://${source}`
          }
        }
      ])
    )

    expect(restored?.target.previewKind).toBe('text')
  })

  it.each([
    ['/work/notes.docx', 'document'],
    ['/work/notes.odt', 'document'],
    ['/work/book.xlsx', 'spreadsheet'],
    ['/work/book.ods', 'spreadsheet'],
    ['/work/legacy.xls', 'spreadsheet'],
    ['/work/people.csv', 'spreadsheet'],
    ['/work/deck.pptx', 'presentation']
  ])('upgrades a persisted binary %s tab', (source, previewKind) => {
    const [restored] = decodePreviewTabs(
      JSON.stringify([
        {
          id: `file:file://${source}`,
          target: {
            binary: true,
            kind: 'file',
            label: source.split('/').pop(),
            path: source,
            previewKind: 'binary',
            source,
            url: `file://${source}`
          }
        }
      ])
    )

    expect(restored?.target.previewKind).toBe(previewKind)
  })
})
