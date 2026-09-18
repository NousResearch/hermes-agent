import { useCallback } from 'react'

import { isEmojiIndexLoaded, searchEmoji } from '@/lib/emoji-index'

import { type CompletionEntry, type CompletionPayload, useLiveCompletionAdapter } from './use-live-completion-adapter'

/**
 * `:shortcode:` completions for the composers, Slack-style (`:joy` → 😂).
 *
 * The catalog itself lives in `lib/emoji-index` (shared with the session stamp
 * picker, so both rank and match the same way). Every query is answered from
 * memory after the first load, so `isCached` skips the debounce and loading
 * state once that has landed.
 *
 * A pick inserts the emoji CHARACTER as plain text — not a chip. Directive
 * chips exist to carry machine-readable references the backend resolves
 * (@file:, /skill); a picked emoji is just text, so it rides the formatter's
 * `rawText` path and lands inline.
 */

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
    isCached: isEmojiIndexLoaded,
    toItem
  })
}
