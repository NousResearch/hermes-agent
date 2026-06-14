/**
 * Memoizing wrapper around `rehype-katex`.
 *
 * Why: the default `@streamdown/math` plugin runs `rehype-katex` on every
 * markdown commit. During streaming, that means each new token re-runs
 * KaTeX on EVERY math node in the message — including equations that
 * haven't changed since the last token. For math-heavy responses (a
 * model deriving an equation step-by-step) this becomes a major source
 * of jank: 20 unchanged equations each pay ~5–20ms of katex.renderToString
 * work per token, adding up to hundreds of ms of CPU bound work that
 * delays the next streaming update.
 *
 * What this plugin does: walk the hast tree looking for the math nodes
 * that `remark-math` emits (`<code class="math-inline">…</code>` for
 * inline and `<pre><code class="math-display">…</code></pre>` for
 * display), key them by `(displayMode, value)`, and serve them from an
 * in-memory LRU cache when we've rendered the same equation before.
 * Cache misses still go through `katex.renderToString`; cache hits
 * return the previously generated hast subtree.
 *
 * Result: each unique equation only pays the katex cost once. Adding
 * one new equation to a paragraph re-renders just that one equation
 * instead of all of them. The cache is process-global so it survives
 * moves between messages (e.g., re-rendering a session).
 *
 * Compatibility: the produced hast structure matches what `rehype-katex`
 * itself produces — we use the same `hast-util-from-html-isomorphic`
 * fragment parsing and the same parent-splice semantics, including the
 * `<pre>`-walk-up for display mode. Drop-in replacement for the math
 * slot in streamdown's PluginConfig.
 *
 * Wire it in via `createMemoizedMathPlugin`:
 *
 *   import { createMemoizedMathPlugin } from '@/lib/katex-memo'
 *   const math = createMemoizedMathPlugin({ singleDollarTextMath: true })
 *   <Streamdown plugins={{ math }} ... />
 */

import type { Element, ElementContent, Parent, Root } from 'hast'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { toText } from 'hast-util-to-text'
import katex from 'katex'
import remarkMath from 'remark-math'
import type { Pluggable } from 'unified'
import { SKIP, visitParents } from 'unist-util-visit-parents'
import type { VFile } from 'vfile'

interface KatexMemoOptions {
  /**
   * Color used for KaTeX errors when we fall back to the lenient parser.
   * Mirrors `@streamdown/math`'s default so the visual output is identical.
   */
  errorColor?: string
}

interface MathPluginConfig {
  /**
   * Match `singleDollarTextMath` from `@streamdown/math`. When true the
   * remark-math parser treats `$x$` as inline math; when false it requires
   * `$$x$$`. Models almost always emit the single-dollar form, so we
   * default it to true at the createMemoizedMathPlugin call site.
   */
  singleDollarTextMath?: boolean
  errorColor?: string
}

/** Cached rendered hast — children to splice into the math node's parent. */
type CachedRender = ElementContent[]

const CACHE_LIMIT = 512

class LruCache<K, V> {
  private readonly map = new Map<K, V>()

  get(key: K): undefined | V {
    const value = this.map.get(key)

    if (value === undefined) {
      return undefined
    }

    // Refresh recency by re-inserting at the tail. Map iteration order is
    // insertion order, so the oldest entry is at the head.
    this.map.delete(key)
    this.map.set(key, value)

    return value
  }

  set(key: K, value: V): void {
    if (this.map.has(key)) {
      this.map.delete(key)
    } else if (this.map.size >= CACHE_LIMIT) {
      const oldest = this.map.keys().next().value

      if (oldest !== undefined) {
        this.map.delete(oldest)
      }
    }

    this.map.set(key, value)
  }
}

const cache = new LruCache<string, CachedRender>()

function cacheKey(displayMode: boolean, value: string): string {
  // `\u0001` is a control character that (a) won't appear in normal
  // markdown and (b) is a single byte so the join is cheap.
  return `${displayMode ? 'd' : 'i'}\u0001${value}`
}

/**
 * Render one math expression with the same two-pass strategy `rehype-katex`
 * uses internally: try strict first (so genuine TeX errors get reported in
 * the VFile message stream), and on failure fall back to lenient mode so
 * the document still renders without a thrown exception. The lenient
 * fallback paints the equation in `errorColor` instead of erroring out.
 */
function renderMath(
  value: string,
  displayMode: boolean,
  errorColor: string,
  file: VFile,
  element: Element
): ElementContent[] {
  let html: string

  try {
    html = katex.renderToString(value, { displayMode, throwOnError: true })
  } catch (error) {
    const cause = error as Error

    file.message('Could not render math with KaTeX', {
      cause,
      place: element.position,
      ruleId: cause.name?.toLowerCase() ?? 'katex',
      source: 'rehype-katex-memo'
    })

    try {
      html = katex.renderToString(value, {
        displayMode,
        errorColor,
        strict: 'ignore',
        throwOnError: false
      })
    } catch {
      // Last-resort fallback — render the source text inside a styled span
      // so the user at least sees what was supposed to be there. Mirrors
      // rehype-katex's own escape hatch.
      return [
        {
          type: 'element',
          tagName: 'span',
          properties: {
            className: ['katex-error'],
            style: `color:${errorColor}`,
            title: String(error)
          },
          children: [{ type: 'text', value }]
        }
      ]
    }
  }

  const fragment = fromHtmlIsomorphic(html, { fragment: true })

  return fragment.children as ElementContent[]
}

/**
 * The actual rehype plugin. Wraps `rehype-katex`'s logic with our LRU
 * cache. Mirrors the upstream visitor exactly except for the cache lookup
 * and an LRU.set on miss.
 */
function createMemoizedRehypeKatex(options: KatexMemoOptions = {}): Pluggable {
  const errorColor = options.errorColor ?? 'var(--color-muted-foreground)'

  return () =>
    function transform(tree: Root, file: VFile): undefined {
      visitParents(tree, 'element', (element, parents) => {
        const classes = Array.isArray(element.properties?.className) ? (element.properties.className as string[]) : []

        // Match the same class set rehype-katex looks for. `language-math`
        // is the markdown ` ```math ` form, `math-inline` is what
        // remark-math emits for `$x$`, `math-display` for `$$x$$`.
        const languageMath = classes.includes('language-math')
        const mathDisplay = classes.includes('math-display')
        const mathInline = classes.includes('math-inline')

        if (!(languageMath || mathDisplay || mathInline)) {
          return
        }

        let displayMode = mathDisplay
        let scope: Element = element
        let parent: Parent | undefined = parents[parents.length - 1]

        // For ` ```math ` the scope walks up to the wrapping <pre> and
        // we treat it as display math. Same logic rehype-katex uses.
        if (languageMath && parent && parent.type === 'element' && (parent as Element).tagName === 'pre') {
          scope = parent as Element
          parent = parents[parents.length - 2]
          displayMode = true
        }

        // No parent means the math node is at the root — there's nothing
        // to splice into, so bail. This shouldn't happen for properly
        // nested markdown but is the same defensive guard rehype-katex has.
        if (!parent) {
          return
        }

        const value = toText(scope, { whitespace: 'pre' })
        const key = cacheKey(displayMode, value)
        let cached = cache.get(key)

        if (!cached) {
          cached = renderMath(value, displayMode, errorColor, file, scope)
          cache.set(key, cached)
        }

        // Splice CLONES of the cached children into the parent. Reusing
        // the same node instances across renders would let downstream
        // rehype plugins or toJsxRuntime mutate the cached subtree —
        // breaking the next cache hit. structuredClone is ~100µs per
        // equation, well below the ~5–20ms katex.renderToString cost
        // we're avoiding.
        const clonedChildren = cached.map(child => structuredClone(child))
        const index = parent.children.indexOf(scope as ElementContent)

        if (index === -1) {
          return
        }

        parent.children.splice(index, 1, ...clonedChildren)

        return SKIP
      })
    }
}

/**
 * Build a streamdown MathPlugin object that uses the memoized rehype-katex
 * wrapper. Drop-in for `@streamdown/math`'s `createMathPlugin`.
 */
export function createMemoizedMathPlugin(config: MathPluginConfig = {}) {
  const remarkPlugin: Pluggable = [remarkMath, { singleDollarTextMath: config.singleDollarTextMath ?? false }]

  const rehypePlugin = createMemoizedRehypeKatex({ errorColor: config.errorColor })

  return {
    name: 'katex' as const,
    type: 'math' as const,
    remarkPlugin,
    rehypePlugin,
    getStyles: () => 'katex/dist/katex.min.css'
  }
}
