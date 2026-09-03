import { describe, expect, it } from 'vitest'

import {
  normalizeScreenTutorAnnotations,
  normalizeScreenTutorPoint,
  screenTutorPointInDisplay,
  screenTutorThumbnailSize,
  selectScreenTutorSource
} from './screen-tutor'

describe('normalizeScreenTutorAnnotations', () => {
  it('normalizes supported primitives and clamps the lifetime', () => {
    expect(
      normalizeScreenTutorAnnotations({
        annotations: [
          { color: 'amber', kind: 'arrow', label: ' Breakout ', x: 0.1, x2: 0.8, y: 0.7, y2: 0.2 },
          { kind: 'label', label: 'Support', x: 0.25, y: 0.75 }
        ],
        displayId: '7',
        frozen: true,
        guide: {
          id: 'excel-pivot',
          instruction: 'Open the Insert tab',
          step: 1,
          successCheck: 'The Insert ribbon is visible',
          title: 'Build a pivot table',
          total: 4
        },
        ttlMs: 999_999
      })
    ).toEqual({
      annotations: [
        { color: 'amber', kind: 'arrow', label: 'Breakout', x: 0.1, x2: 0.8, y: 0.7, y2: 0.2 },
        { color: 'cyan', kind: 'label', label: 'Support', x: 0.25, y: 0.75 }
      ],
      displayId: '7',
      frozen: true,
      guide: {
        id: 'excel-pivot',
        instruction: 'Open the Insert tab',
        step: 1,
        successCheck: 'The Insert ribbon is visible',
        title: 'Build a pivot table',
        total: 4
      },
      mode: 'replace',
      ttlMs: 300_000
    })
  })

  it('drops malformed primitives and rejects an empty result', () => {
    expect(
      normalizeScreenTutorAnnotations({ annotations: [{ kind: 'line', x: 0.1, y: 0.2 }], displayId: '7' })
    ).toBeNull()
    expect(normalizeScreenTutorAnnotations({ annotations: [], displayId: '7' })).toBeNull()
  })
})

describe('screenTutorThumbnailSize', () => {
  it('caps a scaled desktop at the configured long edge', () => {
    expect(screenTutorThumbnailSize({ height: 1440, width: 2560 }, 1.5)).toEqual({ height: 1152, width: 2048 })
  })

  it('does not upscale a small desktop', () => {
    expect(screenTutorThumbnailSize({ height: 600, width: 800 }, 1)).toEqual({ height: 600, width: 800 })
  })
})

describe('selectScreenTutorSource', () => {
  it('matches Electron display ids exactly', () => {
    const sources = [{ display_id: '11' }, { display_id: '22' }]

    expect(selectScreenTutorSource(sources, '22')).toBe(sources[1])
    expect(selectScreenTutorSource(sources, '33')).toBeNull()
  })

  it('accepts Electron builds without display_id only for a single source', () => {
    const source = {}

    expect(selectScreenTutorSource([source], '22')).toBe(source)
  })
})

describe('normalizeScreenTutorPoint', () => {
  it('accepts normalized coordinates and bounds the label', () => {
    expect(normalizeScreenTutorPoint({ displayId: '7', label: ' Save ', x: 0.25, y: 0.75 })).toEqual({
      displayId: '7',
      label: 'Save',
      x: 0.25,
      y: 0.75
    })
  })

  it('rejects missing displays and out-of-range coordinates', () => {
    expect(normalizeScreenTutorPoint({ displayId: '', x: 0.1, y: 0.2 })).toBeNull()
    expect(normalizeScreenTutorPoint({ displayId: '7', x: 1.1, y: 0.2 })).toBeNull()
  })
})

describe('screenTutorPointInDisplay', () => {
  it('maps normalized coordinates in display-local DIP space', () => {
    expect(screenTutorPointInDisplay({ x: 0.5, y: 0.25 }, { height: 1080, width: 1920 })).toEqual({ x: 960, y: 270 })
  })
})
