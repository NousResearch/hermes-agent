/**
 * Large-paste-to-attachment policy.
 *
 * Pasting more than ~3k characters converts the content into a text
 * attachment instead of inserting it inline, keeping the composer clean and
 * preventing a single paste from flooding the input. Short pastes stay inline;
 * the threshold lives here so every paste handler shares one policy.
 */

/** Characters beyond which a plain-text paste becomes a `.txt` attachment. */
export const LARGE_PASTE_ATTACHMENT_THRESHOLD = 3_000

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

/**
 * How long an inline-paste chord (⌘⇧V / Ctrl+Shift+V) stays armed. The chord
 * fires a native `paste` event that carries no modifier state, so the keydown
 * arms a timestamp and the paste handler consumes it. The window only needs to
 * outlast keydown → paste dispatch; a keydown with nothing to paste expires
 * instead of leaking into the next ⌘V.
 */
export const INLINE_PASTE_SHORTCUT_WINDOW_MS = 1_000

/**
 * True for ⌘⇧V (macOS) / Ctrl+Shift+V: "paste as plain text", which here means
 * insert the clipboard text inline regardless of the large-paste threshold —
 * the way a slash command (`/goal …`) needs its material inside the prompt,
 * since attachments can't ride along with slash commands or skills.
 */
export function isInlinePasteShortcut(event: KeyboardEvent): boolean {
  return (event.metaKey || event.ctrlKey) && event.shiftKey && !event.altKey && event.key.toLowerCase() === 'v'
}

/** True when an inline-paste chord armed at `requestedAt` is still live at `now`. */
export function isInlinePasteRequested(requestedAt: number, now: number = Date.now()): boolean {
  return requestedAt > 0 && now - requestedAt < INLINE_PASTE_SHORTCUT_WINDOW_MS
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
