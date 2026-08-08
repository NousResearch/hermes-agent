/**
 * Large-paste-to-attachment policy.
 *
 * Inspired by ChatGPT Work's composer behavior (OpenAI release notes,
 * Aug 4 2026): pasting more than ~10k characters converts the content into a
 * text attachment instead of inserting it inline, keeping the composer clean
 * and preventing a single paste from flooding the input. Short pastes stay
 * inline; the threshold lives here so every paste handler shares one policy.
 */

/** Characters beyond which a plain-text paste becomes a `.txt` attachment. */
export const LARGE_PASTE_ATTACHMENT_THRESHOLD = 10_000

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
