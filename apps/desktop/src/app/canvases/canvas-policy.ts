import type { CanvasDefinition } from './types'

export const CANVAS_SCHEMA_VERSION = '1.0' as const

export const CANVAS_MANIFEST_MAX_BYTES = 460_000

export function canvasRefreshPrompt(canvas: CanvasDefinition, manifestPath: string): string {
  return `Execute the Canvas refresh for "${canvas.id}" now. This is a file operation, not a request for an explanation.

Read the existing manifest at ${manifestPath}; it is this report's contract.

Preserve id, schemaVersion, profile, data sources, and the general structure. Update only data, metrics, insights, dates, and data-dependent blocks. Do not generate HTML as the primary source, include CSS, invent sources, or change the schema.

Required execution steps: read ${manifestPath}; refresh only declared sources; set updatedAt to the current ISO-8601 timestamp; write valid JSON to a temporary sibling; validate it; atomically replace exactly ${manifestPath}; reread the final file and calculate SHA-256.

Never claim success merely because you produced JSON in chat. If any step fails, leave the manifest unchanged and return CANVAS_WRITE_FAILED with the reason. Your final response must contain exactly: CANVAS_WRITE_OK path=${manifestPath} updatedAt=<ISO-8601> sha256=<64-hex>. Only emit it after verifying the final file.

Original request: ${canvas.source.prompt || 'Not available'}
Canvas rules: ${canvas.source.instructions || 'Keep the existing contract.'}`
}

export function canvasRepairPrompt(manifestPath: string): string {
  return `Repair the Canvas manifest at ${manifestPath} now. It is too large for Hermes Desktop to read remotely.

This is a file operation, not an explanation. Read the manifest, preserve its id, title, profile, intent, sources, and useful aggregated data, then rewrite a valid hermes.canvas/v1 manifest smaller than ${CANVAS_MANIFEST_MAX_BYTES} bytes. Design a compact generic document from the request: keep totals, useful summaries, charts, and a relevant bounded table/list. Do not embed an unbounded database dump or opaque HTML. Keep full-data retrieval instructions in source.instructions for a later refresh.

Write to a temporary sibling, validate JSON and byte size, atomically replace exactly ${manifestPath}, reread it, and calculate SHA-256. If successful, reply exactly: CANVAS_WRITE_OK path=${manifestPath} updatedAt=<ISO-8601> sha256=<64-hex>. Otherwise: CANVAS_WRITE_FAILED reason=<reason>.`
}
