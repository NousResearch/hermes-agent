import { mediaKind } from '@/lib/media'

const imageSourceCache = new WeakMap<object, { input: unknown; output: unknown; sources: string[] }>()

/** References, not fetchable URLs: local paths must go through the scoped file bridge.
 * Streaming re-prices and re-groups unchanged tool payloads on each text delta;
 * keep large stdout scans out of that hot path without retaining old messages. */
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
  const args = input && typeof input === 'object' ? (input as Record<string, unknown>) : {}
  const result = output && typeof output === 'object' ? (output as Record<string, unknown>) : {}
  const sources = new Set<string>()

  const add = (value: unknown, explicitImage = false) => {
    if (typeof value !== 'string' || !value.trim()) {
      return
    }

    const source = value.trim()

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

  // Prefer the pixels actually supplied to the model (including crops) over the original input.
  const content = Array.isArray(result.content) ? result.content : []

  for (const block of content) {
    if (!block || typeof block !== 'object') {
      continue
    }

    if (block.type === 'image_url') {
      add(block.image_url?.url, true)
    }

    if (block.type === 'image' && typeof block.data === 'string' && /^image\//.test(block.mimeType ?? '')) {
      add(`data:${block.mimeType};base64,${block.data}`)
    }
  }

  if (sources.size) {
    return [...sources]
  }

  const meta = result.meta && typeof result.meta === 'object' ? (result.meta as Record<string, unknown>) : {}
  add(meta.screenshot_path)

  for (const key of ['screenshot_path', 'host_image', 'image_url', 'image_path', 'image', 'path', 'url']) {
    add(result[key], key === 'image_url')
  }

  // Browser Use emits this explicit marker in stdout; never scrape arbitrary page image URLs.
  for (const key of ['output', 'stdout', 'text', 'text_summary', 'result']) {
    const text = result[key]

    if (typeof text !== 'string') {
      continue
    }

    for (const match of text.matchAll(/^[\t ]*Screenshot path:[\t ]*(.+)$/gm)) {
      add(match[1])
    }

    // MCP image blocks are materialized as standalone MEDIA markers by the adapter.
    for (const match of text.matchAll(/^[\t ]*MEDIA:[\t ]*(.+)$/gm)) {
      add(match[1])
    }
  }

  if (sources.size) {
    return [...sources]
  }

  for (const key of ['image_url', 'image_path', 'path']) {
    add(args[key], key === 'image_url')
  }

  return [...sources]
}
