import { describe, expect, it } from 'vitest'

import {
  ANNOTATION_COLORS,
  ANNOTATION_TOOLS,
  type AnnotationShape,
  colorName,
  DEFAULT_ANNOTATION_COLOR,
  isClick,
  nextCalloutNumber,
  nextShapeId,
  shapeBounds,
  shapesToLegend,
  toolLabel
} from './annotation-model'

function shape(overrides: Partial<AnnotationShape> = {}): AnnotationShape {
  return {
    id: 'a',
    tool: 'arrow',
    color: '#ef4444',
    points: [{ x: 0, y: 0 }],
    ...overrides
  }
}

describe('annotation-model', () => {
  it('exposes the five required tools (rect, ellipse, arrow, pen, callout)', () => {
    expect(ANNOTATION_TOOLS).toEqual(['rect', 'ellipse', 'arrow', 'pen', 'callout'])
  })

  it('defaults the color to red and offers preset swatches', () => {
    expect(DEFAULT_ANNOTATION_COLOR).toBe('#ef4444')
    expect(ANNOTATION_COLORS).toHaveLength(6)
    expect(ANNOTATION_COLORS).toContain(DEFAULT_ANNOTATION_COLOR)
  })

  it('generates unique shape ids', () => {
    expect(nextShapeId()).not.toBe(nextShapeId())
  })

  it('computes shape bounds from its points', () => {
    const bounds = shapeBounds(shape({ points: [{ x: 10, y: 20 }, { x: 30, y: 5 }] }))
    expect(bounds).toEqual({ minX: 10, minY: 5, maxX: 30, maxY: 20 })
  })

  it('returns zero bounds for a shapeless record', () => {
    expect(shapeBounds(shape({ points: [] }))).toEqual({ minX: 0, minY: 0, maxX: 0, maxY: 0 })
  })

  it('treats a close point pair as a click and a far pair as a drag', () => {
    expect(isClick([{ x: 0, y: 0 }, { x: 2, y: 2 }])).toBe(true)
    expect(isClick([{ x: 0, y: 0 }, { x: 50, y: 50 }])).toBe(false)
    expect(isClick([{ x: 0, y: 0 }])).toBe(true)
  })

  it('assigns ascending callout badge numbers', () => {
    expect(nextCalloutNumber([])).toBe(1)
    expect(
      nextCalloutNumber([
        shape({ tool: 'callout', number: 1 }),
        shape({ tool: 'arrow' }),
        shape({ tool: 'callout', number: 3 })
      ])
    ).toBe(4)
  })

  it('maps tools and colors to legend words', () => {
    expect(toolLabel('rect')).toBe('rectangle')
    expect(toolLabel('pen')).toBe('freehand')
    expect(colorName('#22c55e')).toBe('green')
    expect(colorName('#123456')).toBe('#123456')
  })

  it('builds an ordered legend with callout numbers included', () => {
    const legend = shapesToLegend([
      shape({ tool: 'arrow', color: '#ef4444' }),
      shape({ tool: 'callout', color: '#3b82f6', number: 1 })
    ])

    expect(legend).toBe('Region 1: red arrow\nRegion 2: blue callout 1')
  })

  it('builds an empty legend from no shapes', () => {
    expect(shapesToLegend([])).toBe('')
  })
})
