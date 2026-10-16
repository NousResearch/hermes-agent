import { parseMaybeObject } from './fallback-model/format'
import type { ToolPart } from './fallback-model/types'

export type InspectionSectionId = 'args' | 'result' | 'metadata' | 'command' | 'stdout' | 'stderr' | 'diff'

export interface InspectionSection {
  id: InspectionSectionId
  value: unknown
}

// Bound even a single enormous line. Paging affects only painting, never the
// selected value, search range or clipboard payload.
export const INSPECTION_PAGE_CHARS = 16_000

export function inspectionSections(part: ToolPart, inlineDiff = ''): InspectionSection[] {
  const args = parseMaybeObject(part.args)
  const result = parseMaybeObject(part.result)

  const sections: InspectionSection[] = [
    { id: 'args', value: part.args },
    { id: 'result', value: part.result },
    {
      id: 'metadata',
      value: {
        toolName: part.toolName,
        toolCallId: part.toolCallId,
        timestamp: part.timestamp,
        completedAt: part.completedAt,
        isError: part.isError,
        // Optional, for the lossless live-result projection. Older transcripts
        // have only the result, which remains inspectable without this field.
        display: 'toolResultMetadata' in part ? part.toolResultMetadata : undefined
      }
    }
  ]

  const command = typeof args.command === 'string' ? args.command : args.code

  if (typeof command === 'string') {
    sections.push({ id: 'command', value: command })
  }

  for (const id of ['stdout', 'stderr'] as const) {
    if (typeof result[id] === 'string') {
      sections.push({ id, value: result[id] })
    }
  }

  const diff =
    typeof result.inline_diff === 'string'
      ? result.inline_diff
      : typeof result.diff === 'string'
        ? result.diff
        : inlineDiff

  if (diff) {
    sections.push({ id: 'diff', value: diff })
  }

  return sections
}

/** Only the selected section is serialized, after the inspector is opened. */
export function inspectionText(value: unknown): string | undefined {
  if (typeof value === 'string') {
    return value
  }

  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return undefined
  }
}

export function inspectionWindow(text: string, offset: number) {
  let start = Math.max(0, Math.min(offset, Math.max(0, text.length - 1)))
  let end = Math.min(text.length, start + INSPECTION_PAGE_CHARS)

  // Do not paint half a surrogate pair at either edge of the window.
  if (start > 0 && /[\uDC00-\uDFFF]/.test(text[start]) && /[\uD800-\uDBFF]/.test(text[start - 1])) {
    start--
  }

  if (end < text.length && /[\uD800-\uDBFF]/.test(text[end - 1]) && /[\uDC00-\uDFFF]/.test(text[end])) {
    end++
  }

  return { start, end, text: text.slice(start, end) }
}
