/**
 * Pop-out annotate-flush relay (main process side).
 *
 * The popped-out Browser lives in its own OS window/renderer with no composer,
 * so it cannot flush comment pins locally. It packages its pin stack and posts
 * the envelope here; the main process validates the shape and forwards it to
 * the primary window, whose renderer attaches the crops + prompt to the real
 * composer. Keeps the untrusted renderer payload small and well-shaped before
 * it crosses the IPC boundary.
 */

export interface AnnotateFlushItem {
  imageDataUrl?: string
  note?: string
  number?: number
  prompt?: string
}

export interface AnnotateFlushEnvelope {
  id?: string
  items?: AnnotateFlushItem[]
  pageUrl?: string
}

/** Hard ceiling on pins per flush — a human click, not a bulk import. */
export const ANNOTATE_FLUSH_MAX_ITEMS = 100

/** Hard ceiling on the encoded crops riding one flush (~base64 PNG bytes). */
export const ANNOTATE_FLUSH_MAX_CHARS = 32 * 1024 * 1024

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isFlushItem(value: unknown): value is AnnotateFlushItem {
  if (!isRecord(value)) {
    return false
  }

  if (typeof value.number !== 'number' || !Number.isFinite(value.number)) {
    return false
  }

  if (typeof value.prompt !== 'string') {
    return false
  }

  if (value.imageDataUrl !== undefined && typeof value.imageDataUrl !== 'string') {
    return false
  }

  if (value.note !== undefined && typeof value.note !== 'string') {
    return false
  }

  return true
}

/** Shape + size gate for a renderer-posted flush envelope. */
export function isAnnotateFlushEnvelope(value: unknown): value is AnnotateFlushEnvelope {
  if (!isRecord(value)) {
    return false
  }

  if (typeof value.id !== 'string' || value.id.length === 0 || value.id.length > 256) {
    return false
  }

  if (!Array.isArray(value.items) || value.items.length === 0 || value.items.length > ANNOTATE_FLUSH_MAX_ITEMS) {
    return false
  }

  if (value.pageUrl !== undefined && typeof value.pageUrl !== 'string') {
    return false
  }

  let chars = 0

  for (const item of value.items) {
    if (!isFlushItem(item)) {
      return false
    }

    chars += (item.imageDataUrl?.length ?? 0) + item.prompt.length + (item.note?.length ?? 0)

    if (chars > ANNOTATE_FLUSH_MAX_CHARS) {
      return false
    }
  }

  return true
}
