/**
 * The bundled emojibase catalog — ONE loader and ONE search for every surface
 * that offers emoji (the composer's `:shortcode:` completions, the session
 * stamp picker). Two copies would drift in ranking and in which labels match.
 *
 * Draws from the emojibase-data package served at `./emojibase` by the
 * `hermes:emojibase-assets` vite plugin — offline, no CDN, because Electron has
 * to work without a network. The index loads lazily on the first query and every
 * later query is answered from memory, so a caller can skip its debounce and
 * loading state with `isEmojiIndexLoaded()` once that first load has landed.
 *
 * Entries are the emoji CHARACTER, not an image: the glyph is drawn by the
 * platform's own emoji font (Apple Color Emoji on macOS, Segoe UI Emoji on
 * Windows), which is what makes a stamp look native on the machine it is read
 * on.
 */

export interface EmojiEntry {
  emoji: string
  /** The emoji's primary shortcode, e.g. "joy". */
  code: string
  /** Every shortcode, tag, and label a search may match. */
  haystack: string[]
}

const EMOJIBASE_DIR = './emojibase'

let indexPromise: Promise<EmojiEntry[]> | null = null
let indexLoaded = false

/** True once the catalog is in memory — a caller's own debounce/loading state
 *  exists only for that first load. */
export const isEmojiIndexLoaded = (): boolean => indexLoaded

async function fetchIndex(): Promise<EmojiEntry[]> {
  const [dataRes, codesRes] = await Promise.all([
    fetch(`${EMOJIBASE_DIR}/en/data.json`),
    fetch(`${EMOJIBASE_DIR}/en/shortcodes/emojibase.json`)
  ])

  const data: { emoji: string; hexcode: string; label: string; tags?: string[] }[] = await dataRes.json()
  const codes: Record<string, string | string[]> = await codesRes.json()
  const entries: EmojiEntry[] = []

  for (const item of data) {
    const raw = codes[item.hexcode]

    if (!raw) {
      continue
    }

    const shortcodes = Array.isArray(raw) ? raw : [raw]

    entries.push({
      emoji: item.emoji,
      code: shortcodes[0],
      haystack: [...shortcodes, ...(item.tags ?? []), item.label.toLowerCase()]
    })
  }

  indexLoaded = true

  return entries
}

/** The catalog, loaded once per renderer session. A failed load is NOT cached:
 *  the next query retries rather than leaving every later search empty. */
export function loadEmojiIndex(): Promise<EmojiEntry[]> {
  if (!indexPromise) {
    indexPromise = fetchIndex().catch(error => {
      indexPromise = null

      throw error
    })
  }

  return indexPromise
}

/** Prefix matches on shortcodes rank first, then tag/label substring hits. */
export async function searchEmoji(query: string, limit = 8): Promise<EmojiEntry[]> {
  const index = await loadEmojiIndex()
  const q = query.toLowerCase()
  const prefix: EmojiEntry[] = []
  const loose: EmojiEntry[] = []

  for (const entry of index) {
    if (entry.code.startsWith(q) || entry.haystack.some(h => h.startsWith(q))) {
      prefix.push(entry)
    } else if (entry.haystack.some(h => h.includes(q))) {
      loose.push(entry)
    }

    if (prefix.length >= limit) {
      break
    }
  }

  return [...prefix, ...loose].slice(0, limit)
}
