/**
 * Render cost of one message's content parts, in budget units.
 *
 * Two layers bound long transcripts and both spend the same currency: the
 * store window (how many messages reach assistant-ui at all) and the DOM page
 * budget (how many of those actually render). Neither can be a message COUNT —
 * counting only parts underpriced a 51KB tool result as "1", so a handful of
 * huge results let a 600KB transcript through the old 300-part cap and drove
 * Chromium's renderer into a GC crash. Characters approximate markdown
 * parsing, text-node allocation, and tool-result formatting; parts approximate
 * component/node count.
 *
 * Shared so a heavy-but-short session is bounded by the same rule as a
 * long-but-light one (#55191).
 */

export const RENDER_WEIGHT_CHARS = 512

// Stop traversing once a single message has enough text to consume a complete
// DOM render page. Going further cannot change which whole turn crosses any
// budget, and avoiding an unbounded walk matters for deeply nested tool
// payloads.
const MAX_MEASURED_MESSAGE_CHARS = 300 * RENDER_WEIGHT_CHARS

const contentWeightCache = new WeakMap<object, number>()
const NON_RENDERED_CONTENT_FIELDS = new Set(['id', 'role', 'toolCallId', 'toolName', 'type'])

/**
 * Estimate the synchronous renderer cost of one message's content array.
 *
 * A WeakMap keeps settled history O(message count) on later store updates;
 * both assistant-ui and the session store publish a new content array when a
 * streaming message changes, so the live tail still receives a fresh weight.
 */
export function messageRenderWeight(content: unknown): number {
  if (!Array.isArray(content)) {
    return 1
  }

  const cached = contentWeightCache.get(content)

  if (cached !== undefined) {
    return cached
  }

  const seen = new WeakSet<object>()
  const pending: unknown[] = [...content]
  let characters = 0

  while (pending.length > 0 && characters < MAX_MEASURED_MESSAGE_CHARS) {
    const value = pending.pop()

    if (typeof value === 'string') {
      characters += Math.min(value.length, MAX_MEASURED_MESSAGE_CHARS - characters)

      continue
    }

    if (!value || typeof value !== 'object' || seen.has(value)) {
      continue
    }

    seen.add(value)

    if (Array.isArray(value)) {
      for (const nested of value) {
        pending.push(nested)
      }

      continue
    }

    for (const [key, nested] of Object.entries(value)) {
      if (!NON_RENDERED_CONTENT_FIELDS.has(key)) {
        pending.push(nested)
      }
    }
  }

  const weight = Math.max(1, content.length) + Math.ceil(characters / RENDER_WEIGHT_CHARS)
  contentWeightCache.set(content, weight)

  return weight
}
