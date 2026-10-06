/**
 * Bound the nesting depth of raw inline HTML before it reaches `rehype-raw`.
 *
 * Streamdown parses assistant markdown with `allowDangerousHtml` + `rehype-raw`,
 * so any `<tag>` run in the text is handed to parse5. `hast-util-from-parse5`
 * then walks the resulting tree with mutual recursion (`element → one → all`),
 * one JS frame per level of nesting — so a deeply nested run of UNCLOSED tags
 * overflows the call stack and throws `RangeError: Maximum call stack size
 * exceeded` out of the middle of a React render.
 *
 * That is not a hypothetical: a degenerating model emits thousands of `<unk>`
 * tokens as its reasoning, every one of which parse5 opens as another element
 * that never closes. Measured against our pinned deps, the throw starts at
 * ~1,750 consecutive unclosed tags and the payload persists to the session, so
 * every reload re-renders it — the crash repeats until the message is deleted.
 *
 * The bound is on DEPTH, not size: 20,000 balanced `<b>x</b>` pairs (160KB) and
 * 20,000 void `<br>` tags parse fine, because neither drives the tree deeper.
 * Only unclosed elements accumulate depth, so only they are counted.
 *
 * Past the cap the remaining opening tags are escaped (`<` → `&lt;`) rather than
 * dropped: the text stays visible and readable as literal `<unk>`, which is
 * exactly what it is. Content under the cap is returned untouched — the common
 * case pays one scan and no rewrite.
 */

// parse5 nests only known HTML elements meaningfully, but an unknown tag
// (`<unk>`) still becomes an element node, so the depth cost is identical.
// Well under the ~1,750 measured failure point, and far above any nesting real
// markup uses — a deeply structured HTML document is a few dozen levels.
const MAX_HTML_DEPTH = 300

// Elements that never take children, so an opening tag adds no depth.
const VOID_ELEMENTS = new Set([
  'area',
  'base',
  'br',
  'col',
  'embed',
  'hr',
  'img',
  'input',
  'link',
  'meta',
  'param',
  'source',
  'track',
  'wbr'
])

// `<tag`, `</tag`, or `<tag/`. Deliberately loose on attributes: we only need
// the name and whether it opens, closes, or self-closes.
const TAG_RE = /<(\/?)([a-zA-Z][a-zA-Z0-9-]*)((?:"[^"]*"|'[^']*'|[^>"'])*)>/g

/**
 * Escape raw HTML tags past `MAX_HTML_DEPTH` levels of unclosed nesting.
 *
 * Returns `text` unchanged when the depth never exceeds the cap, so the
 * identity is preserved for the overwhelming majority of messages (which
 * matters: the markdown pipeline caches on string identity).
 */
export function clampHtmlNestingDepth(text: string): string {
  // Fast path: no tags at all, nothing to bound.
  if (!text.includes('<')) {
    return text
  }

  const open: string[] = []
  let depth = 0
  let firstOverflow = -1
  let match: RegExpExecArray | null

  TAG_RE.lastIndex = 0

  while ((match = TAG_RE.exec(text)) !== null) {
    const closing = match[1] === '/'
    const name = match[2].toLowerCase()
    const selfClosing = match[3].endsWith('/')

    if (closing) {
      // Close the nearest matching open element, mirroring parse5's recovery:
      // an unmatched close tag is ignored rather than unwinding the stack.
      const index = open.lastIndexOf(name)

      if (index !== -1) {
        depth = index
        open.length = index
      }

      continue
    }

    if (selfClosing || VOID_ELEMENTS.has(name)) {
      continue
    }

    open.push(name)
    depth += 1

    if (depth > MAX_HTML_DEPTH && firstOverflow === -1) {
      firstOverflow = match.index
    }
  }

  if (firstOverflow === -1) {
    return text
  }

  // Everything before the overflow parses at a safe depth and is left exactly
  // as-is; from there on, opening tags become visible literal text.
  return text.slice(0, firstOverflow) + text.slice(firstOverflow).replaceAll('<', '&lt;')
}
