/**
 * Pure annotation model for the composer image-annotation overlay.
 *
 * Everything in this module is DOM-free and unit-testable: shape records,
 * tool/color presets, pointer→shape math, and legend generation. The React
 * canvas layer (`annotation-canvas.tsx`) is a thin adapter over these shapes.
 */

export type AnnotationTool = 'rect' | 'ellipse' | 'arrow' | 'pen' | 'callout'

export interface Point {
  x: number
  y: number
}

export interface AnnotationShape {
  id: string
  tool: AnnotationTool
  color: string
  /** rect/ellipse/arrow: two points (start, end). pen: polyline. callout: center. */
  points: Point[]
  /** Auto-assigned 1-based badge number for `callout` shapes. */
  number?: number
}

export const ANNOTATION_TOOLS: readonly AnnotationTool[] = ['rect', 'ellipse', 'arrow', 'pen', 'callout']

export const ANNOTATION_COLORS: readonly string[] = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#a855f7', '#111827']

export const DEFAULT_ANNOTATION_COLOR = ANNOTATION_COLORS[0]!

let shapeCounter = 0

export function nextShapeId(): string {
  shapeCounter += 1

  return `annotation-${Date.now().toString(36)}-${shapeCounter}`
}

/** Bounding box of a shape's points (in canvas coordinates). */
export function shapeBounds(shape: AnnotationShape): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const point of shape.points) {
    minX = Math.min(minX, point.x)
    minY = Math.min(minY, point.y)
    maxX = Math.max(maxX, point.x)
    maxY = Math.max(maxY, point.y)
  }

  if (shape.points.length === 0) {
    return { minX: 0, minY: 0, maxX: 0, maxY: 0 }
  }

  return { minX, minY, maxX, maxY }
}

/** True when two points are close enough to count as a click (callout placement). */
export function isClick(points: Point[], tolerance = 4): boolean {
  if (points.length < 2) {
    return true
  }

  const [first, last] = [points[0]!, points[points.length - 1]!]

  return Math.abs(first.x - last.x) <= tolerance && Math.abs(first.y - last.y) <= tolerance
}

/** Number of existing callouts in a shape list (for the next badge number). */
export function nextCalloutNumber(shapes: readonly AnnotationShape[]): number {
  return shapes.reduce((max, shape) => (shape.tool === 'callout' ? Math.max(max, shape.number ?? 0) : max), 0) + 1
}

const TOOL_LABELS: Record<AnnotationTool, string> = {
  rect: 'rectangle',
  ellipse: 'ellipse',
  arrow: 'arrow',
  pen: 'freehand',
  callout: 'callout'
}

const COLOR_NAMES: Record<string, string> = {
  '#ef4444': 'red',
  '#f59e0b': 'orange',
  '#22c55e': 'green',
  '#3b82f6': 'blue',
  '#a855f7': 'purple',
  '#111827': 'black'
}

export function toolLabel(tool: AnnotationTool): string {
  return TOOL_LABELS[tool]
}

export function colorName(color: string): string {
  return COLOR_NAMES[color.toLowerCase()] ?? color
}

/**
 * Auto-generated legend text for the shapes on the canvas. Each shape gets a
 * region number in creation order; the legend is inserted into the composer
 * draft (editable before send) so the model receives the markup semantics
 * alongside the annotated image, e.g.:
 *
 *   Region 1: red arrow
 *   Region 2: blue callout
 */
export function shapesToLegend(shapes: readonly AnnotationShape[]): string {
  const lines = shapes.map((shape, index) => {
    const color = colorName(shape.color)
    const tool = toolLabel(shape.tool)

    if (shape.tool === 'callout' && shape.number !== undefined) {
      return `Region ${index + 1}: ${color} callout ${shape.number}`
    }

    return `Region ${index + 1}: ${color} ${tool}`
  })

  return lines.join('\n')
}
