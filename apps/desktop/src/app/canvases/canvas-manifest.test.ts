import { describe, expect, it } from 'vitest'

import { parseCanvasManifest } from './canvas-manifest'

describe('parseCanvasManifest', () => {
  it('renders arbitrary scalar groups and object listings without a domain schema', () => {
    const manifest = parseCanvasManifest(
      JSON.stringify({
        schema: 'hermes.canvas/v1',
        id: 'directory',
        title: 'Directory',
        source: { prompt: 'Build a directory' },
        data: {
          totals: { all: 2, active: 1 },
          byRegion: { north: 1, south: 1 },
          entries: [
            { id: 1, name: 'Alpha', active: true },
            { id: 2, name: 'Beta', active: false }
          ]
        },
        updatedAt: '2026-07-15T00:00:00Z'
      }),
      'default'
    )

    expect(manifest.blocks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'kpis' }),
        expect.objectContaining({ title: 'By Region', type: 'bar-chart' }),
        expect.objectContaining({ columns: ['Id', 'Name', 'Active'], title: 'Entries', type: 'table' })
      ])
    )
  })

  it('supports the generic document primitives', () => {
    const manifest = parseCanvasManifest(
      JSON.stringify({
        schema: 'hermes.canvas/v1',
        id: 'document',
        title: 'Document',
        source: {},
        document: {
          sections: [
            {
              title: 'Overview',
              elements: [
                { type: 'list', title: 'Actions', items: ['Call', 'Email'] },
                { type: 'image', dataUri: 'data:image/png;base64,AA==', alt: 'Logo' },
                { type: 'divider' }
              ]
            }
          ]
        }
      }),
      'default'
    )

    expect(manifest.blocks.map(block => block.type)).toEqual(['list', 'image', 'divider'])
  })
})
