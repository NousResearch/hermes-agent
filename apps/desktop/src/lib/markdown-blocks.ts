import { parseMarkdownIntoBlocks } from '@assistant-ui/react-streamdown'

/**
 * Block splitting for the streaming markdown pipeline, without re-lexing the
 * whole message on every token flush.
 *
 * `parseMarkdownIntoBlocks` is a full `marked` lex of the entire text —
 * measured 3.4–9.6ms per call at 64–192KB. During streaming every flush is a
 * new string, so the stock splitter pays that O(full-text) cost ~30×/s on
 * long replies. Two caches remove it:
 *
 * 1. Exact-string LRU — a message that REMOUNTS with unchanged text
 *    (virtualizer scroll, session switch) reuses its parse outright.
 * 2. Streaming-append cache — when the new text starts with a recently parsed
 *    text (the token-append case), the previous parse's blocks are reused up
 *    to a settled boundary and only the suffix is lexed. The boundary drops
 *    the previous parse's trailing whitespace-only blocks AND its last content
 *    block, because appended text can retroactively change how that last
 *    block parses (open fence, list/table continuation, setext underline, a
 *    lazy blockquote line). Blocks before it are separated by settled blank
 *    lines and cannot be affected. Cross-block reference links can't regress:
 *    Streamdown renders each block as an independent markdown document
 *    already. Verified property: `blocks.join('') === text`, and incremental
 *    output is asserted byte-identical to a full lex in tests across fences,
 *    lists, tables, setext headings, blockquotes, and HTML blocks.
 *
 * Any doubt — no prefix match, reconstruction mismatch — falls back to the
 * full lex, i.e. exactly the previous behavior.
 */

const EXACT_CACHE_MAX = 64
const EXACT_CACHE_MIN_LENGTH = 1024
const exactCache = new Map<string, string[]>()

// Streaming messages grow monotonically, and only a handful stream at once
// (main reply + reasoning part, maybe a tile). A tiny ring is enough; each
// entry holds the last parse for one growing text lineage.
const APPEND_CACHE_MAX = 4
const APPEND_CACHE_MIN_LENGTH = 2048
const appendCache: { blocks: string[]; text: string }[] = []

function rememberAppend(text: string, blocks: string[]): void {
  if (text.length < APPEND_CACHE_MIN_LENGTH) {
    return
  }

  // Replace the lineage this text grew from (its cached prefix), else push.
  const index = appendCache.findIndex(entry => text.startsWith(entry.text))

  if (index !== -1) {
    appendCache.splice(index, 1)
  }

  appendCache.push({ blocks, text })

  if (appendCache.length > APPEND_CACHE_MAX) {
    appendCache.shift()
  }
}

function lexIncrementally(text: string): null | string[] {
  const entry = appendCache.find(cached => text.length > cached.text.length && text.startsWith(cached.text))

  if (!entry) {
    return null
  }

  // Settled boundary: drop the last TWO content blocks (skipping any
  // whitespace-only blocks around them). Dropping only the single last content
  // block is unsound: appended text can retroactively merge the previous
  // parse's last two blocks into one. The trigger is a trailing Setext
  // underline — `marked` only treats `-`/`=` as an underline for the paragraph
  // ABOVE it, so a settled `"#e\n5\n-"` lexes as ["#e\n", "5\n-"], but growing
  // the tail to `"#e\n5\n-p2=kj:c"` collapses both into one paragraph. The
  // block before the last is the deepest an append can reach (the underline
  // consumes exactly one preceding block), so re-lexing the last two is safe;
  // earlier blocks are fenced off by settled blank lines. join('') === text
  // still holds either way, so the reconstruction check below can't catch this.
  let keep = entry.blocks.length

  for (let dropped = 0; dropped < 2 && keep > 0; dropped += 1) {
    while (keep > 0 && !entry.blocks[keep - 1].trim()) {
      keep -= 1
    }

    if (keep > 0) {
      keep -= 1
    }
  }

  if (keep === 0) {
    return null
  }

  const settled = entry.blocks.slice(0, keep)
  let settledLength = 0

  for (const block of settled) {
    settledLength += block.length
  }

  // Defensive reconstruction check — the splitter's join(blocks) === text
  // property is what makes offsets exact. If it ever doesn't hold, full lex.
  if (settledLength > entry.text.length || !text.startsWith(entry.text.slice(0, settledLength), 0)) {
    return null
  }

  return [...settled, ...parseMarkdownIntoBlocks(text.slice(settledLength))]
}

export function parseMarkdownIntoBlocksCached(markdown: string): string[] {
  if (markdown.length < EXACT_CACHE_MIN_LENGTH) {
    return parseMarkdownIntoBlocks(markdown)
  }

  const hit = exactCache.get(markdown)

  if (hit) {
    // Refresh recency (Map iteration order is insertion order).
    exactCache.delete(markdown)
    exactCache.set(markdown, hit)

    return hit
  }

  const blocks = lexIncrementally(markdown) ?? parseMarkdownIntoBlocks(markdown)

  rememberAppend(markdown, blocks)
  exactCache.set(markdown, blocks)

  if (exactCache.size > EXACT_CACHE_MAX) {
    exactCache.delete(exactCache.keys().next().value as string)
  }

  return blocks
}
