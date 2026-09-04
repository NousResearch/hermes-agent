const EMPTY_ATTACHMENT_REFS: string[] = []

export function partText(part: unknown): string {
  if (typeof part === 'string') {
    return part
  }

  if (!part || typeof part !== 'object') {
    return ''
  }

  const row = part as { text?: unknown; type?: unknown }

  return (!row.type || row.type === 'text') && typeof row.text === 'string' ? row.text : ''
}

export function messageContentText(content: unknown): string {
  if (typeof content === 'string') {
    return content.trim()
  }

  return Array.isArray(content) ? content.map(partText).join('').trim() : ''
}

// Cheap streaming-stable "does this message have visible text" check: returns
// on the first non-whitespace text part without concatenating the whole
// message. Used as a useAuiState selector so its boolean output stays stable
// across token flushes (flips false→true once per turn).
export function contentHasVisibleText(content: unknown): boolean {
  if (typeof content === 'string') {
    return content.trim().length > 0
  }

  if (!Array.isArray(content)) {
    return false
  }

  for (const part of content) {
    if (partText(part).trim().length > 0) {
      return true
    }
  }

  return false
}

export function messageAttachmentRefs(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return EMPTY_ATTACHMENT_REFS
  }

  return value.every(ref => typeof ref === 'string') ? value : EMPTY_ATTACHMENT_REFS
}

export function pickPrimaryPreviewTarget(targets: string[]): string[] {
  if (targets.length <= 1) {
    return targets
  }

  const localUrl = targets.find(value => /^https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])/i.test(value))

  return [localUrl || targets[targets.length - 1]]
}

const LINK_PREVIEW_URL_RE = /https?:\/\/[^\s<>"')\]]+/gi

/**
 * Click-to-expand preview candidates (D7): the external http(s) URLs a
 * settled message mentions, deduped, in first-appearance order. Local hosts
 * are excluded here — the preview bridge refuses them server-side anyway,
 * and a chip for a URL that can never unfurl is noise. Streaming-safe: the
 * caller only runs this once the turn settles.
 */
export function extractLinkPreviewTargets(text: string): string[] {
  if (!text) {
    return []
  }

  const seen = new Set<string>()
  const targets: string[] = []

  for (const match of text.matchAll(LINK_PREVIEW_URL_RE)) {
    const raw = match[0].replace(/[.,;:!]+$/, '')

    if (seen.has(raw)) {
      continue
    }

    seen.add(raw)

    let url: URL

    try {
      url = new URL(raw)
    } catch {
      continue
    }

    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      continue
    }

    if (/^(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?$/i.test(url.host)) {
      continue
    }

    targets.push(raw)
  }

  return targets
}
