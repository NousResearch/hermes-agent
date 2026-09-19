import { mediaKind } from '@/lib/media'

export const TOOL_IMAGE_PAGE_SIZE = 5

const imageSourceCache = new WeakMap<object, { input: unknown; output: unknown; sources: string[] }>()
const IMAGE_FIELDS = ['screenshot_path', 'host_image', 'image_url', 'image_path', 'image', 'path', 'url']

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

// Normalize only the transport roots, not JSON embedded in external text.
function payloadRecord(value: unknown): Record<string, unknown> {
  if (isRecord(value)) {
    return value
  }

  if (typeof value === 'string' && value.trimStart().startsWith('{')) {
    try {
      const parsed: unknown = JSON.parse(value)

      if (isRecord(parsed)) {
        return parsed
      }
    } catch {
      // Incomplete live JSON is not yet a structured image payload.
    }
  }

  return {}
}

/** References, not fetchable URLs: local paths must go through the scoped file bridge.
 * Streaming re-prices and re-groups unchanged tool payloads on each text delta;
 * keep large stdout scans out of that hot path without retaining old messages.
 * Payloads are immutable snapshots: changes must publish a new input/output object.
 * Cache BEFORE parsing JSON; string-only calls have no global cache retaining payloads. */
export function toolImageSources(input: unknown, output: unknown): string[] {
  const key = output && typeof output === 'object' ? output : input && typeof input === 'object' ? input : null
  const cached = key ? imageSourceCache.get(key) : undefined

  if (cached && cached.input === input && cached.output === output) {
    return cached.sources
  }

  const sources = collectToolImageSources(input, output)

  if (key) {
    imageSourceCache.set(key, { input, output, sources })
  }

  return sources
}

function collectToolImageSources(input: unknown, output: unknown): string[] {
  const args = payloadRecord(input)
  const result = payloadRecord(output)
  // Exactly one MCP result envelope. Never walk arbitrary objects or repeated result chains.
  const results = isRecord(result.result) && result.result !== result ? [result, result.result] : [result]
  const sources = new Set<string>()

  const add = (value: unknown, explicitImage = false) => {
    if (typeof value !== 'string' || !value.trim()) {
      return
    }

    const source = value.trim()

    // Control characters can disguise URL schemes after browser normalization.
    if (/\p{Cc}/u.test(source)) {
      return
    }

    if (/^data:image\/[\w.+-]+;base64,/i.test(source)) {
      sources.add(source)
    } else if (/^https?:\/\//i.test(source)) {
      if (explicitImage || mediaKind(source) === 'image') {
        sources.add(source)
      }
    } else if (!/^[a-z][\w+.-]*:/i.test(source) || /^(?:file:|[a-z]:[\\/])/i.test(source)) {
      if (mediaKind(source) === 'image') {
        sources.add(source)
      }
    }
  }

  const addNative = (block: Record<string, unknown>) => {
    if (block.type === 'image_url' && isRecord(block.image_url)) {
      add(block.image_url.url, true)
    }

    if (
      block.type === 'image' &&
      typeof block.data === 'string' &&
      typeof block.mimeType === 'string' &&
      /^image\//.test(block.mimeType)
    ) {
      add(`data:${block.mimeType};base64,${block.data}`)
    }
  }

  // Prefer the pixels actually supplied to the model (including crops) over original references.
  for (const record of results) {
    for (const block of Array.isArray(record.content) ? record.content : []) {
      if (isRecord(block)) {
        addNative(block)
      }
    }
  }

  if (sources.size) {
    return [...sources]
  }

  const addFields = (record: Record<string, unknown>, keys: readonly string[], imageDescriptor = false) => {
    for (const key of keys) {
      const value = record[key]
      add(
        key === 'image_url' && isRecord(value) ? value.url : value,
        key === 'image_url' || (imageDescriptor && key === 'url')
      )
    }
  }

  const addArrays = (record: Record<string, unknown>) => {
    // Arrays have no count cap: pagination bounds mounting, not source discovery.
    for (const image of Array.isArray(record.images) ? record.images : []) {
      if (typeof image === 'string') {
        add(image, true)
      } else if (
        isRecord(image) &&
        (image.type === undefined || image.type === 'image' || image.type === 'image_url')
      ) {
        addNative(image)
        addFields(image, IMAGE_FIELDS, true)
      }
    }

    for (const path of Array.isArray(record.image_paths) ? record.image_paths : []) {
      add(path)
    }
  }

  for (const record of results) {
    if (isRecord(record.meta)) {
      add(record.meta.screenshot_path)
    }

    addFields(record, IMAGE_FIELDS)
    addArrays(record)

    // Only explicit standalone tool markers, never page assets or arbitrary text blocks.
    for (const key of ['output', 'stdout', 'text', 'text_summary', 'result']) {
      const text = record[key]

      if (typeof text !== 'string') {
        continue
      }

      for (const match of text.matchAll(/^[\t ]*Screenshot path:[\t ]*(.+)$/gm)) {
        add(match[1])
      }

      for (const match of text.matchAll(/^[\t ]*MEDIA:[\t ]*(.+)$/gm)) {
        const marker = match[1].trim()
        const quote = marker[0]

        if (quote === '"' || quote === "'" || quote === '`') {
          if (marker.length > 2 && marker.endsWith(quote)) {
            add(marker.slice(1, -1))
          }
        } else {
          add(marker)
        }
      }
    }
  }

  if (!sources.size) {
    addFields(args, ['image_url', 'image_path', 'path'])
    addArrays(args)
  }

  return [...sources]
}
