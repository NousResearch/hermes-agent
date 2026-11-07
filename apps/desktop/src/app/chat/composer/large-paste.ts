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

// Grapheme segmentation, the same idiom as `app/cron/job-state.ts`: a UTF-16 unit
// budget alone can cut an astral character or a ZWJ sequence mid-cluster.
const previewGraphemes = new Intl.Segmenter(undefined, { granularity: 'grapheme' })

/**
 * The paste's title preview: the longest grapheme prefix that fits in `maxChars` UTF-16 units.
 *
 * `slice(0, maxChars)` counts code UNITS, so a cut inside an astral character kept half of the
 * surrogate pair (`…context_engine  <half of 🧩>`). That half survives `JSON.stringify`, rides the
 * `prompt.submit` params as `title_preview`, and then aborts the admission's sqlite bind with
 * `UnicodeEncodeError: surrogates not allowed` — reported to the user as a bare "internal error"
 * with the prompt lost (#150281). Dropping the whole cluster at the boundary costs at most one
 * character of a preview that is only used for titling, and keeps every consumer display-safe.
 */
export function pasteTitlePreview(text: string, maxChars: number = LARGE_PASTE_TITLE_PREVIEW_CHARS): string {
  if (maxChars <= 0) {
    return ''
  }

  if (text.length <= maxChars) {
    return text
  }

  // Stop at the first cluster that does not fit. The iteration is incremental, so the boundary
  // cluster is seen whole (a fixed-size window would cut it and mistake its tail for a complete
  // segment) and the rest of the paste is never segmented: measured 12 ms on a 5.4 MB paste
  // against 312 ms for a full pass.
  let end = 0

  for (const { segment } of previewGraphemes.segment(text)) {
    const next = end + segment.length

    if (next > maxChars) {
      break
    }

    end = next
  }

  return text.slice(0, end)
}

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
