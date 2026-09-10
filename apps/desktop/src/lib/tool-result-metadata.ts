/** Gateway display hints are not the tool's result. Keep them off the raw
 * payload so summaries cannot overwrite evidence, including false/null/"". */
export interface ToolResultMetadata {
  duration_s?: number
  error?: string | boolean
  inline_diff?: string
  message?: string
  preview?: string
  summary?: string
  todos?: unknown
}

export interface ToolResultSource {
  result?: unknown
  toolResultMetadata?: ToolResultMetadata
}

/** A derived record for existing presentation code; never stored as the result. */
export function toolResultRecord(source: ToolResultSource): Record<string, unknown> {
  let value = source.result

  if (typeof value === 'string') {
    try {
      value = JSON.parse(value)
    } catch {
      value = undefined
    }
  }

  const record = value && typeof value === 'object' && !Array.isArray(value) ? value : {}

  // Authoritative result fields win over fallible/abbreviated display hints.
  return { ...source.toolResultMetadata, ...record }
}
