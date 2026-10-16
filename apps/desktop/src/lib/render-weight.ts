import { isCardTool, isFileEditTool, isSilentTool } from '@/lib/tool-render-class'

/**
 * Render cost of one message's content parts, in budget units.
 *
 * Two layers bound long transcripts and both spend the same currency: the
 * store window (how many messages reach assistant-ui at all) and the DOM page
 * budget (how many of those actually render). Neither can be a message COUNT —
 * counting only parts underpriced a 51KB tool result as "1", so a handful of
 * huge results let a 600KB transcript through the old 300-part cap and drove
 * Chromium's renderer into a GC crash (#55191). Characters approximate
 * markdown parsing, text-node allocation, and tool-result formatting; parts
 * approximate component/node count.
 *
 * The two layers do NOT price a part the same way, because they protect
 * different things:
 *
 *   - The STORE window protects the heap. Every message it admits is
 *     normalized into the runtime repository whether or not the transcript
 *     collapses it, so it prices the payload it has to hold: `messageStoreWeight`.
 *   - The DOM budget protects the paint, and what a turn paints is decided by
 *     the GROUPING, not by the bytes behind it. A settled run of twelve reads
 *     is one grey summary line, a thought is one collapsed disclosure, a
 *     `todo` is hoisted out of the transcript entirely, and a generated image
 *     is one `<img>` however long its data URL. Charging those their payload
 *     had the budget counting hundreds of units of work that never mounts, so
 *     "Show earlier" appeared after two or three tool-heavy turns of a session
 *     that was painting almost nothing: `messagePaintWeight`.
 */

export const RENDER_WEIGHT_CHARS = 512

// Stop traversing once a single message has enough text to consume a complete
// DOM render page. Going further cannot change which whole turn crosses any
// budget, and avoiding an unbounded walk matters for deeply nested tool
// payloads.
const MAX_MEASURED_MESSAGE_CHARS = 300 * RENDER_WEIGHT_CHARS

const storeWeightCache = new WeakMap<object, number>()
const paintWeightCache = new WeakMap<object, number>()
const NON_RENDERED_CONTENT_FIELDS = new Set(['id', 'role', 'toolCallId', 'toolName', 'type'])

/**
 * What a collapsed row costs the DOM: one line of scaffolding.
 *
 * A settled tool row is an icon, a title and maybe a count; a settled thought
 * is a "Thought for 12s" header. Either one keeps its payload behind a
 * disclosure, and an unopened disclosure mounts none of it. History always
 * mounts collapsed, and history is exactly what "Show earlier" pages back
 * through.
 */
const COLLAPSED_ROW_WEIGHT = 1

/**
 * What a fixed-size card costs the DOM.
 *
 * A generated image is one `<img>` whether its result carries a path or a
 * multi-megabyte data URL; a clarify question is a prompt and a few buttons; a
 * delegation is a header over a one-line ticker. None of them scale with the
 * payload, so charging characters priced a single image at more than a whole
 * page of real turns.
 */
const CARD_WEIGHT = 6

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

/**
 * Character cost of an arbitrary payload, bounded and cycle-safe.
 *
 * `budget` is the characters still worth measuring. It is threaded through a
 * whole message rather than reset per part, so a message of many huge parts
 * cannot walk past the ceiling one part at a time.
 */
function payloadCharacters(roots: readonly unknown[], budget: number): number {
  const seen = new WeakSet<object>()
  const pending: unknown[] = [...roots]
  let characters = 0

  while (pending.length > 0 && characters < budget) {
    const value = pending.pop()

    if (typeof value === 'string') {
      characters += Math.min(value.length, budget - characters)

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

  return characters
}

/** Payload price: one unit per part, plus one per 512 characters it carries. */
function payloadWeight(parts: readonly unknown[], budget: number): number {
  return parts.length + Math.ceil(payloadCharacters(parts, budget) / RENDER_WEIGHT_CHARS)
}

/**
 * Estimate the cost of holding one message's content array in the runtime.
 *
 * A WeakMap keeps settled history O(message count) on later store updates;
 * both assistant-ui and the session store publish a new content array when a
 * streaming message changes, so the live tail still receives a fresh weight.
 */
export function messageStoreWeight(content: unknown): number {
  if (!Array.isArray(content)) {
    return 1
  }

  const cached = storeWeightCache.get(content)

  if (cached !== undefined) {
    return cached
  }

  const weight = Math.max(1, payloadWeight(content, MAX_MEASURED_MESSAGE_CHARS))
  storeWeightCache.set(content, weight)

  return weight
}

/**
 * What one part mounts, priced the way `message-parts.tsx` renders it.
 *
 * `measure` prices a payload against the message's shared character ceiling —
 * only the parts that actually paint their content spend from it.
 */
function partPaintWeight(part: unknown, measure: (parts: readonly unknown[]) => number): number {
  if (!isRecord(part)) {
    return 1
  }

  // A thought mounts its header; the reasoning text sits behind it.
  if (part.type === 'reasoning') {
    return COLLAPSED_ROW_WEIGHT
  }

  if (part.type !== 'tool-call') {
    // Text is markdown the DOM really builds, so it keeps the payload price.
    return measure([part])
  }

  const toolName = typeof part.toolName === 'string' ? part.toolName : ''

  if (isSilentTool(toolName)) {
    return 0
  }

  if (!isCardTool(toolName)) {
    return COLLAPSED_ROW_WEIGHT
  }

  // A diff is the one card that scales: `FileDiffPanel` mounts a row per line,
  // and a big patch really is the expensive thing in the turn.
  return isFileEditTool(toolName) ? measure([part]) : CARD_WEIGHT
}

/**
 * Estimate what one message's content array actually MOUNTS in the transcript.
 *
 * Cached like the store weight: a settled message keeps its number across
 * later store updates, and a streaming one publishes a fresh array per delta
 * so the live tail is always re-measured.
 */
export function messagePaintWeight(content: unknown): number {
  if (!Array.isArray(content)) {
    return 1
  }

  const cached = paintWeightCache.get(content)

  if (cached !== undefined) {
    return cached
  }

  // One character ceiling for the whole message, not one per part — otherwise a
  // message of many huge parts walks past it one part at a time.
  let remaining = MAX_MEASURED_MESSAGE_CHARS

  const measure = (parts: readonly unknown[]) => {
    const characters = payloadCharacters(parts, remaining)
    remaining -= characters

    return parts.length + Math.ceil(characters / RENDER_WEIGHT_CHARS)
  }

  let weight = 0

  for (const part of content) {
    weight += partPaintWeight(part, measure)
  }

  weight = Math.max(1, weight)
  paintWeightCache.set(content, weight)

  return weight
}
