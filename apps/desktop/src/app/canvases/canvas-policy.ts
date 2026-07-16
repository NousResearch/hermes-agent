import type { CanvasDefinition } from './types'

export const CANVAS_SCHEMA_VERSION = '1.0' as const

export function canvasRefreshPrompt(canvas: CanvasDefinition, manifestPath: string): string {
  return `Update Canvas "${canvas.id}".

Read the existing manifest at ${manifestPath}; it is this report's contract.

Preserve id, schemaVersion, profile, data sources, and the general structure. Update only data, metrics, insights, dates, and data-dependent blocks. Do not generate HTML as the primary source, include CSS, invent sources, or change the schema.

Write a valid temporary file first, validate the JSON, and replace the manifest only after completion. Preserve the previous version if an error occurs.

Original request: ${canvas.source.prompt || 'Not available'}
Canvas rules: ${canvas.source.instructions || 'Keep the existing contract.'}`
}
