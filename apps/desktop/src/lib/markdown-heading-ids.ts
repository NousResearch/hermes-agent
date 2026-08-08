/**
 * Heading-id stamping + in-page anchor resolution for the Desktop
 * markdown preview pane (#81055).
 *
 * Streamdown does not ship a heading slugger: `<h2>` reaches the DOM
 * without an `id`, so a same-document `#fragment` link — which Streamdown
 * also renders as a `<button>` with no `href` — has no anchor target and
 * no navigation affordance. We stamp the `id` ourselves via a rehype
 * plugin (after Streamdown's raw → sanitize → harden chain so `id` is not
 * stripped by rehype-sanitize's default attribute allow-list) and resolve
 * `#hash` clicks against the rendered headings.
 *
 * Slug shape follows github-slugger so the resulting anchors match
 * GitHub / Obsidian / any rehype-slug-based renderer. Duplicate slugs are
 * disambiguated the same way (`slug`, `slug-1`, `slug-2`).
 */

const HEADING_SELECTOR = 'h1, h2, h3, h4, h5, h6'

/** GitHub-style slug: lowercase, strip non-letter/digit/mark/dash/underscore/space. */
export function githubSlug(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\p{M}\-_\s]/gu, '')
    .replace(/\s/g, '-')
}

/** Decoration-insensitive key for tolerant matching only. */
function looseKey(value: string): string {
  return value
    .toLowerCase()
    .replace(/\p{No}/gu, '')
    .replace(/[^\p{L}\p{N}]/gu, '')
}

/** Drop a leading section number (`3-`, `1.2)`, `§4 `) from a slug. */
function withoutLeadingNumber(value: string): string {
  return value.replace(/^[\s§#]*\d+(?:[.\-)]\d+)*[.\-)\s]*/, '')
}

interface HastNode {
  children?: HastNode[]
  properties?: Record<string, unknown>
  tagName?: string
  type?: string
  value?: string
}

function nodeText(node: HastNode): string {
  if (node.type === 'text') {
    return node.value || ''
  }
  // `<br>` in a heading is a line break, not a word-boundary remover.
  if (node.tagName === 'br') {
    return ' '
  }
  return (node.children || []).map(nodeText).join('')
}

/**
 * rehype plugin: stamp an `id` on every heading from its text, deduplicating
 * collisions the way github-slugger does (`slug`, `slug-1`, `slug-2`).
 *
 * Any existing id is replaced on purpose: this runs after rehype-harden,
 * which has already rewritten author-supplied raw-HTML ids to
 * `user-content-*` — an address no generated TOC uses.
 */
export function rehypeHeadingIds() {
  return (tree: HastNode) => {
    const counts = new Map<string, number>()
    const visit = (node: HastNode) => {
      if (node.type === 'element' && /^h[1-6]$/.test(node.tagName || '')) {
        const base = githubSlug(nodeText(node))
        if (base) {
          const seen = counts.get(base) ?? 0
          counts.set(base, seen + 1)
          node.properties = {
            ...(node.properties || {}),
            id: seen === 0 ? base : `${base}-${seen}`,
          }
        }
      }
      for (const child of node.children || []) {
        visit(child)
      }
    }
    visit(tree)
  }
}

/** Decode a raw `#...` href into the bare fragment it addresses. */
export function hashFragment(href: string): string {
  const raw = href.replace(/^#/, '')
  try {
    return decodeURIComponent(raw)
  } catch {
    return raw
  }
}

/**
 * Resolve a `#hash` against the headings rendered in `container`: exact id
 * first, then decoration-insensitive, then with a leading section number
 * stripped. Returns null when nothing plausibly matches — a dead anchor
 * should stay dead rather than scroll somewhere arbitrary.
 */
export function findHeadingByHash(
  container: ParentNode,
  href: string
): HTMLElement | null {
  const fragment = hashFragment(href)
  if (!fragment) {
    return null
  }
  const headings = Array.from(
    container.querySelectorAll<HTMLElement>(HEADING_SELECTOR)
  )
  const idOf = (element: HTMLElement) => element.getAttribute('id') || ''

  const exact = headings.find((element) => idOf(element) === fragment)
  if (exact) {
    return exact
  }

  const target = looseKey(fragment)
  if (!target) {
    return null
  }
  const loose = headings.find((element) => looseKey(idOf(element)) === target)
  if (loose) {
    return loose
  }

  const stripped = looseKey(withoutLeadingNumber(fragment))
  if (!stripped) {
    return null
  }
  return (
    headings.find(
      (element) =>
        looseKey(withoutLeadingNumber(idOf(element))) === stripped
    ) ||
    headings.find((element) => looseKey(idOf(element)) === stripped) ||
    null
  )
}