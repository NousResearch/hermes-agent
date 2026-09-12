import { describe, expect, it } from 'vitest'

import { $previewTabs, commitBrowserTabLocation, decodePreviewTabs, setBrowserTabPageZoom } from './preview'

describe('persisted preview migration', () => {
  it('persists page zoom per browser tab across navigation and defaults invalid values to actual size', () => {
    const tabs = decodePreviewTabs(
      JSON.stringify([
        {
          id: 'url:browser-one',
          pageZoomPercent: 125,
          target: { kind: 'url', label: 'One', source: 'https://one.example', url: 'https://one.example' }
        },
        {
          id: 'url:browser-two',
          pageZoomPercent: 'huge',
          target: { kind: 'url', label: 'Two', source: 'https://two.example', url: 'https://two.example' }
        }
      ])
    )

    expect(tabs.map(tab => tab.pageZoomPercent)).toEqual([125, 100])

    $previewTabs.set(tabs)
    setBrowserTabPageZoom('url:browser-two', 140)
    setBrowserTabPageZoom('url:browser-two', Number.NaN)
    commitBrowserTabLocation('url:browser-two', 'https://two.example/next', 'Next')

    const restored = decodePreviewTabs(window.localStorage.getItem('hermes.desktop.previewTabs.v2') ?? '[]')

    expect(restored.map(tab => [tab.id, tab.pageZoomPercent, tab.target.url])).toEqual([
      ['url:browser-one', 125, 'https://one.example'],
      ['url:browser-two', 140, 'https://two.example/next']
    ])
  })

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
})
