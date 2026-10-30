import { useCallback } from 'react'

import { type CompletionEntry, type CompletionPayload, useLiveCompletionAdapter } from './use-live-completion-adapter'

/**
 * `:shortcode:` completions for the composers, Slack-style (`:joy` → 😂).
 *
 * Draws from the same bundled emojibase-data the reaction picker uses (served
 * at ./emojibase by the `hermes:emojibase-assets` vite plugin — offline, no
 * CDN). The index lazy-loads on the first `:` trigger, then every query is
 * answered from memory, so `isCached` skips the debounce and loading state
 * after that first load.
 *
 * A pick inserts the emoji CHARACTER as plain text — not a chip. Directive
 * chips exist to carry machine-readable references the backend resolves
 * (@file:, /skill); a picked emoji is just text, so it rides the formatter's
 * `rawText` path and lands inline.
 */

interface EmojiEntry {
  emoji: string
  /** Primary shortcode, e.g. "joy". */
  code: string
  /** Every shortcode, tag, and label that should match a search. */
  haystack: string[]
}

let indexPromise: Promise<EmojiEntry[]> | null = null
let indexLoaded = false

async function loadIndex(): Promise<EmojiEntry[]> {
  const [dataRes, codesRes] = await Promise.all([
    fetch('./emojibase/en/data.json'),
    fetch('./emojibase/en/shortcodes/emojibase.json')
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

/** Prefix matches on shortcodes rank first, then tag/label substring hits. */
async function searchEmoji(query: string, limit = 8): Promise<EmojiEntry[]> {
  const index = await (indexPromise ??= loadIndex())
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

export function useEmojiCompletions() {
  const fetcher = useCallback(async (query: string): Promise<CompletionPayload> => {
    const entries = await searchEmoji(query)

    return {
      query,
      items: entries.map(entry => ({
        text: entry.emoji,
        display: `${entry.emoji}  :${entry.code}:`,
        meta: ''
      }))
    }
  }, [])

  const toItem = useCallback(
    (entry: CompletionEntry, index: number) => ({
      id: `${entry.text}|${index}`,
      type: 'emoji',
      label: typeof entry.display === 'string' ? entry.display : entry.text,
      metadata: {
        display: typeof entry.display === 'string' ? entry.display : entry.text,
        // The formatter's serialize() returns rawText verbatim → the emoji
        // character lands as plain inline text, no chip.
        rawText: entry.text,
        meta: '',
        group: '',
        action: ''
      }
    }),
    []
  )

  return useLiveCompletionAdapter({
    enabled: true,
    fetcher,
    isCached: () => indexLoaded,
    toItem
  })
}
