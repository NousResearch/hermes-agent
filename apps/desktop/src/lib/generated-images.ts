type ToolLike = {
  result?: unknown
  toolName?: unknown
  type?: unknown
}

type TextLike = {
  text?: unknown
  type?: unknown
}

// Path-ish result fields the model may echo into its prose. Display prefers the
// host path (gateway-deliverable); stripping must catch every variant so a
// sandbox path the model restated doesn't slip through as a duplicate image.
const DISPLAY_KEYS = ['host_image', 'image'] as const
const ECHO_KEYS = ['host_image', 'image', 'agent_visible_image'] as const

function recordFromUnknown(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }

  if (typeof value !== 'string' || !value.trim()) {
    return null
  }

  try {
    const parsed = JSON.parse(value)

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : null
  } catch {
    return null
  }
}

function stringFields(record: Record<string, unknown>, keys: readonly string[]): string[] {
  return keys.map(key => record[key]).filter((v): v is string => typeof v === 'string' && v.trim().length > 0)
}

function regexEscape(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))]
}

function imageResult(part: ToolLike): Record<string, unknown> | null {
  if (part.type !== 'tool-call' || part.toolName !== 'image_generate') {
    return null
  }

  const record = recordFromUnknown(part.result)

  return record && record.success !== false ? record : null
}

/** Display source for a completed `image_generate` result (host path wins). */
export function generatedImageFromResult(result: unknown): string | null {
  const record = recordFromUnknown(result)

  if (!record || record.success === false) {
    return null
  }

  return stringFields(record, DISPLAY_KEYS)[0] ?? null
}

/** Every path/URL a generated image might appear as in prose, for de-duping. */
export function generatedImageEchoSources(parts: readonly ToolLike[]): string[] {
  return unique(parts.flatMap(part => stringFields(imageResult(part) ?? {}, ECHO_KEYS)))
}

/** Strip a generated image out of prose so it only ever shows in the tool slot.
 *  Once a generation succeeded (`sources` is non-empty) we drop every embedded
 *  image and media link from that message — the model frequently restates the
 *  remote URL while the result holds the local path, so matching the exact
 *  source is not enough. Bare occurrences of the known paths/URLs are removed
 *  too. Surrounding prose is preserved. */
export function stripGeneratedImageEchoes(text: string, sources: readonly string[]): string {
  if (!text || sources.length === 0) {
    return text
  }

  let next = text
    .replace(/!\[[^\]\n]*\]\([^)\n]*\)/g, '')
    .replace(/\[[^\]\n]*\]\(\s*#media:[^)\n]*\)/g, '')

  for (const source of unique([...sources])) {
    next = next.replace(new RegExp(String.raw`(^|[\s([{])<?${regexEscape(source)}>?(?=$|[\s)\]},.!?])`, 'g'), '$1')
  }

  return next
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]{2,}/g, ' ')
    .trim()
}

/** Strip generated-image echoes from text parts, dropping any part left empty.
 *  The image lives in the tool slot; prose keeps the agent's actual words. */
export function dedupeGeneratedImageEchoesInParts<T extends TextLike & ToolLike>(parts: readonly T[]): T[] {
  const sources = generatedImageEchoSources(parts)

  if (!sources.length) {
    return [...parts]
  }

  return parts
    .map(part =>
      part.type === 'text' && typeof part.text === 'string'
        ? { ...part, text: stripGeneratedImageEchoes(part.text, sources) }
        : part
    )
    .filter(part => part.type !== 'text' || (typeof part.text === 'string' && part.text.trim().length > 0))
}
