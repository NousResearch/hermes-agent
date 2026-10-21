// Shared position lookups over `s.thread.messages`, built once per array.
//
// Every mounted message row re-runs its `useAuiState` selectors on each thread
// notify — one per streamed chunk. A selector that scans the transcript to find
// its own row therefore costs mounted-rows x transcript-length per chunk, which
// is how streaming into a long chat got slower with every settled turn
// (#126486). The runtime publishes one messages array per notify, so keying on
// its identity lets every row in that notify share a single O(n) pass.

type Row = { readonly id: string; readonly role: string }

const indexCache = new WeakMap<readonly Row[], Map<string, number>>()
const userOrdinalCache = new WeakMap<readonly Row[], Map<string, number>>()

function memo(
  cache: WeakMap<readonly Row[], Map<string, number>>,
  messages: readonly Row[],
  build: () => Map<string, number>
) {
  let map = cache.get(messages)

  if (!map) {
    map = build()
    cache.set(messages, map)
  }

  return map
}

/** Index of the LAST row with `id`, or -1. */
export function threadMessageIndex(messages: readonly Row[], id: string): number {
  return (
    memo(indexCache, messages, () => {
      const map = new Map<string, number>()

      messages.forEach((message, index) => map.set(message.id, index))

      return map
    }).get(id) ?? -1
  )
}

/** Position of `id` among the user rows (first occurrence), or null. */
export function threadUserOrdinal(messages: readonly Row[], id: string): null | number {
  return (
    memo(userOrdinalCache, messages, () => {
      const map = new Map<string, number>()

      for (const message of messages) {
        if (message.role === 'user' && !map.has(message.id)) {
          map.set(message.id, map.size)
        }
      }

      return map
    }).get(id) ?? null
  )
}
