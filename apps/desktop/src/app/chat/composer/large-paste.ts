/**
 * Large-paste-to-attachment policy.
 *
 * By default, pasting more than 3k characters converts the content into a text
 * attachment instead of inserting it inline, keeping the composer clean and
 * preventing a single paste from flooding the input. Short pastes stay inline;
 * the configurable threshold's default and bounds live here with the policy.
 */

/** Characters beyond which a plain-text paste becomes a `.txt` attachment. */
export const LARGE_PASTE_ATTACHMENT_THRESHOLD = 3_000

export const LARGE_PASTE_ATTACHMENT_THRESHOLD_MIN = 0
export const LARGE_PASTE_ATTACHMENT_THRESHOLD_MAX = 100_000

/** Missing or invalid preferences retain the existing paste behavior. */
export function normalizeLargePasteAttachmentThreshold(value: unknown): number {
  if (typeof value !== 'number' && (typeof value !== 'string' || value.trim() === '')) {
    return LARGE_PASTE_ATTACHMENT_THRESHOLD
  }

  const threshold = Number(value)

  return Number.isInteger(threshold) &&
    threshold >= LARGE_PASTE_ATTACHMENT_THRESHOLD_MIN &&
    threshold <= LARGE_PASTE_ATTACHMENT_THRESHOLD_MAX
    ? threshold
    : LARGE_PASTE_ATTACHMENT_THRESHOLD
}

/** Maximum source text retained exclusively for automatic title generation. */
export const LARGE_PASTE_TITLE_PREVIEW_CHARS = 1_000

/**
 * True when a plain-text paste should be converted into a text attachment
 * rather than inserted inline. Only sheer size qualifies — rich clipboard
 * data, images, and files never route through this path (they have their own
 * pipelines upstream of this check).
 */
export function shouldConvertPasteToAttachment(
  text: string,
  threshold: number = LARGE_PASTE_ATTACHMENT_THRESHOLD
): boolean {
  return typeof text === 'string' && threshold > 0 && text.length > threshold
}

// `electron/composer-paste.ts` names every saved paste `pasted_content_<stamp>_<hex>.txt`;
// staging into the session may append `-N`. Anything else is a real file.
const PASTED_CONTENT_FILE_RE = /(?:^|[\\/])pasted_content_[\w-]+\.txt$/

/** True for a large-paste file, whose chip reads "Pasted content" instead of its path. */
export function isPastedContentPath(path: string): boolean {
  return PASTED_CONTENT_FILE_RE.test(path)
}

/** Human-readable size of a paste's UTF-8 bytes, for the attachment chip. */
export function pasteSizeLabel(text: string): string {
  const bytes = new TextEncoder().encode(text).length

  if (bytes < 1024) {
    return `${bytes} B`
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
