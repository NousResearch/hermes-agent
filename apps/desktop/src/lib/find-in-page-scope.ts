/**
 * Renderer-side find-in-page that scopes matches to the CURRENT VIEW — the
 * active chat surface — instead of the whole document.
 *
 * Background. Electron's `webContents.findInPage` searches the renderer's
 * entire DOM. The desktop chat shell mounts every ever-active chat surface
 * simultaneously (see apps/desktop/AGENTS.md and the keep-alive comment in
 * apps/desktop/src/app/chat/index.tsx), so a global search matches across
 * every conversation, every background tile, and every other pane that
 * happens to render a transcript. That is not what the user expects when
 * they press ⌘F in a chat — they expect to search the chat they are reading
 * (#81726).
 *
 * Strategy. At bar-open time, capture the active chat surface element and
 * remember it for the lifetime of the find bar. Every query highlights the
 * occurrences inside that subtree, ⌘G / ⌘⇧G step between them, and closing the
 * bar drops the highlight. If a keep-alive tab flip hides the captured surface
 * while the bar stays open (the route didn't change, so the FindBar's pathname
 * cleanup never ran), the scope re-resolves to the new foreground surface —
 * "find wherever the user is reading" holds across a flip, not just at open
 * time (#81726).
 *
 * "Current view" is the foreground `[data-chat-surface]` element after
 * filtering out inactive keep-alive tabs. That is the same policy every other
 * document-wide lookup obeys (see pane-visibility.ts), so this can be reasoned
 * about as "find wherever the user is reading".
 *
 * Painting. Matches are painted with the **CSS Custom Highlight API**
 * (`CSS.highlights` + `Range`s), never by inserting nodes. This is a
 * correctness requirement, not an optimization:
 *
 * - The transcript is React-owned and re-renders while the bar is open
 *   (assistant responses stream through `markdown-text.tsx`, which rebuilds the
 *   markdown DOM on every delta). Injecting `<mark>`s into that tree means
 *   fighting the reconciler: a render that re-allocates a region detaches the
 *   marks, so the old engine had to re-walk and re-wrap the WHOLE scope on
 *   every mutation to keep highlights consistent. Ranges are inert — React
 *   never sees them, and a re-render simply re-ranks which ranges are still
 *   attached, which one cheap re-scan fixes.
 * - The scope also contains a live `contentEditable` composer (`chat/index.tsx`).
 *   Splitting and replacing text nodes under a live caret is exactly the kind
 *   of external DOM write the editing machinery is not required to survive.
 * - Cost. Wrapping every occurrence allocates a node per match (plus the text
 *   nodes around it) and forces layout. Measured with the desktop's own
 *   Electron build (Chromium 142) on a deterministic fixture the size the DOM
 *   budget in `thread/list.tsx` allows — 875,824 characters, 41,977 text nodes,
 *   42,281 elements — searching `e` (99,759 occurrences) added 99,769
 *   `<mark>`s, took 1.20-1.36s for the keystroke, 6.1-8.7s for three more
 *   keystrokes, ~0.1-0.3s per Enter/Ctrl+G step, and let only 2-4 streamed
 *   deltas through in 8s (~2-4s per delta: the observer re-wraps the whole
 *   scope on every mutation, so the renderer is saturated for as long as the
 *   assistant streams). The same fixture through `CSS.highlights`: the same
 *   query costs 58ms, a step 0.1ms, 146 deltas land in 8s, and the DOM gains
 *   ten nodes instead of a hundred thousand. The first scan after page load
 *   pays V8 warm-up (~200ms on this fixture); later scans are ~10-60ms.
 *
 * Re-scanning. The scope is watched with a MutationObserver, but a mutation
 * only marks the scan dirty; the re-scan itself is **throttled** to one per
 * {@link RESCAN_MIN_INTERVAL_MS} so a streaming turn cannot starve the main
 * thread, and it never scrolls (a background re-render must not move the
 * reader — see "Never navigate, move focus, or open a surface because
 * something happened in the background" in apps/desktop/AGENTS.md). Stepping
 * with an unchanged, non-dirty scope reuses the existing ranges, so ⌘G costs
 * nothing.
 *
 * Support. The Custom Highlight API is the only painting path; there is no
 * node-inserting fallback, because that fallback IS the freeze this module was
 * rewritten to remove. Where the API is missing (jsdom in the unit tests), the
 * module still counts, steps and scrolls — only the paint is skipped.
 */

import { queryVisible } from '@/components/pane-shell/pane-visibility'

const SCOPE_SELECTOR = '[data-chat-surface]'
const ROOT_ATTR = 'data-find-root'

/** Registry name for every match ("hermes-" prefix: the page is shared with
 *  plugin content, and a bare "find" name is a collision waiting to happen). */
const ALL_HIGHLIGHT = 'hermes-find'
/** Registry name for the match the bar is currently on. Registered AFTER the
 *  all-matches highlight, since `CSS.highlights` paints in insertion order and
 *  the active match must win. */
const ACTIVE_HIGHLIGHT = 'hermes-find-active'

/**
 * Subtrees the walker never looks into.
 *
 * - `script,style,noscript` — neither painted nor searchable.
 * - `[role="search"]` — the find bar's own control; its text must never be a
 *   match of itself.
 * - editors (`[contenteditable]`, `textarea`, `[role="textbox"]`) — the
 *   composer's draft is a live editing surface, not part of the document being
 *   read. Counting it made the match counter disagree with what is on screen
 *   (#134070), and writing into it is what corrupted the caret and the writing
 *   direction there.
 *
 * Skipped subtrees cost nothing per text node: they are collected once per scan
 * (see {@link collectRanges}).
 */
const SKIPPED_SELECTOR =
  'script,style,noscript,[role="search"],[contenteditable]:not([contenteditable="false"]),textarea,[role="textbox"]'

/** At most one mutation-driven re-scan per this interval. A streaming turn
 *  mutates the scope on every delta; a scan is ~15-30ms on a full page, so
 *  unthrottled re-scans would spend the frame budget on highlights. */
const RESCAN_MIN_INTERVAL_MS = 200

/** Above this many skip subtrees, testing each text node against all of them
 *  costs more than the `closest()` walk it replaces. */
const MAX_SKIP_ROOTS = 8

let observer: MutationObserver | null = null
let rescanTimer: null | ReturnType<typeof setTimeout> = null
let scopeRoot: HTMLElement | null = null
let ranges: Range[] = []
let activeIndex = 0
let activeQuery = ''
/** The scope changed since the last scan: a step must re-scan before moving. */
let dirty = false
let lastScanAt = 0

/**
 * The element the open FindBar should search. Captured at bar-open time and
 * reused for every query / step until the bar closes. Reading the visible
 * chat surface here means the scope follows the user's current focus —
 * pressing ⌘F in chat A searches chat A even if the route later flips to
 * chat B before the user finishes typing (a typed query still targets A;
 * route-change cleanup in FindBar closes the bar first).
 */
export function resolveCurrentFindScope(): HTMLElement | null {
  return queryVisible<HTMLElement>(SCOPE_SELECTOR)
}

/** Capture the current view as the find scope. Called when the bar opens. */
export function captureFindScope(): HTMLElement | null {
  const root = resolveCurrentFindScope()

  resetFindState()
  scopeRoot = root

  if (root) {
    root.setAttribute(ROOT_ATTR, '')
  }

  return root
}

/**
 * The scope captured by the open bar, or null if none is active.
 *
 * When the captured surface was hidden by a keep-alive tab flip (the bar only
 * closes on a ROUTE change, so a flip keeps it open over a now-inactive pane),
 * re-resolves to the foreground surface instead of going dead — otherwise
 * every query reports 0/0 and ⌘G becomes a no-op on the surface the user is
 * actually reading (#81726).
 */
export function currentFindScope(): HTMLElement | null {
  const roots = document.querySelectorAll<HTMLElement>(`[${ROOT_ATTR}]`)

  for (const root of roots) {
    if (!isElementInHiddenPane(root)) {
      return root
    }
  }

  if (roots.length === 0) {
    // No captured scope — the bar was never opened, or releaseFindScope
    // already tore it down. Nothing to re-target to.
    return null
  }

  // Every marked root is hidden — the captured surface lost a tab flip while
  // the bar stayed open. Move the scope to the now-foreground surface.
  return retargetFindScope(roots)
}

/**
 * Move the find scope to the foreground chat surface after a keep-alive tab
 * flip hid the captured one. Drops the stale surface's ranges (they belong to
 * a subtree the user can no longer see) and stamps `data-find-root` on the new
 * surface so the walker keeps searching where the user is now reading. Returns
 * the new scope, or null when no chat surface is visible (the user flipped to
 * a non-chat pane — nothing on screen qualifies as a view).
 */
function retargetFindScope(roots: NodeListOf<HTMLElement>): HTMLElement | null {
  const foreground = resolveCurrentFindScope()

  if (!foreground) {
    return null
  }

  for (const root of roots) {
    if (root === foreground) {
      continue
    }

    root.removeAttribute(ROOT_ATTR)
  }

  // The observer is still watching the OLD root, and the ranges point into it;
  // the next performScopedFind re-scans the new scope (its fast path requires
  // live ranges, so the drop cannot be stepped over).
  stopObserver()
  dropHighlights()
  scopeRoot = foreground
  foreground.setAttribute(ROOT_ATTR, '')

  return foreground
}

/** Same predicate pane-visibility exposes, kept local so this module is
 *  independently testable without an import cycle in the renderer. */
function isElementInHiddenPane(element: Element): boolean {
  return Boolean(element.closest('[data-pane-hidden]'))
}

/**
 * Every (case-insensitive) occurrence of `query` in `root`, as a `Range` per
 * occurrence in document order.
 *
 * Two choices here keep the scan cheap on a full page — measured on the same
 * 875,824-character / 41,977-text-node fixture as the module header: 10ms for a
 * query with no matches, 58ms for one with 99,759 of them (the first scan after
 * page load pays V8 warm-up and lands near 200ms). The node-wrapping engine
 * spent 1.2-1.4s on that same query, because it paid for both the scan and
 * 99,769 new DOM nodes.
 *
 * - `SHOW_TEXT` with no `NodeFilter` callback. A filter callback is a JS call
 *   per visited node, element and text alike (~84K crossings on a full page);
 *   the skip subtrees are collected once per scan instead and tested with one
 *   `contains()` per text node — usually zero of them, since the find bar
 *   itself mounts outside `[data-chat-surface]`.
 * - No text-node splitting. A `Range` can start and end anywhere inside a
 *   single text node, so a match costs one object and no DOM write.
 */
function collectRanges(root: Element, query: string): Range[] {
  const lowerQuery = query.toLowerCase()
  const found: Range[] = []

  if (!lowerQuery) {
    return found
  }

  const skipRoots = [...root.querySelectorAll(SKIPPED_SELECTOR)]

  // Pathological page (many skip subtrees): one `closest()` per text node beats
  // testing every node against every subtree.
  const isSkipped =
    skipRoots.length > MAX_SKIP_ROOTS
      ? (node: Node) => Boolean((node as Text).parentElement?.closest(SKIPPED_SELECTOR))
      : (node: Node) => skipRoots.some(skipRoot => skipRoot.contains(node))

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)

  while (walker.nextNode()) {
    const node = walker.currentNode

    if (isSkipped(node)) {
      continue
    }

    const text = node.nodeValue ?? ''
    const lowerText = text.toLowerCase()
    let index = lowerText.indexOf(lowerQuery)

    while (index !== -1) {
      const range = document.createRange()

      range.setStart(node, index)
      range.setEnd(node, index + lowerQuery.length)
      found.push(range)

      index = lowerText.indexOf(lowerQuery, index + lowerQuery.length)
    }
  }

  return found
}

/** The registry, or null where the Custom Highlight API is unavailable. */
function highlightRegistry(): HighlightRegistry | null {
  const css = (globalThis as { CSS?: { highlights?: HighlightRegistry } }).CSS

  return typeof Highlight === 'function' && css?.highlights ? css.highlights : null
}

/** Publish every match. Deleting before setting keeps the active highlight
 *  last in insertion order, which is what makes it paint on top. */
function publishRanges(): void {
  const registry = highlightRegistry()

  if (!registry) {
    return
  }

  registry.delete(ALL_HIGHLIGHT)
  registry.delete(ACTIVE_HIGHLIGHT)

  if (ranges.length === 0) {
    return
  }

  const all = new Highlight()

  for (const range of ranges) {
    all.add(range)
  }

  registry.set(ALL_HIGHLIGHT, all)
  publishActive()
}

/**
 * Move the active highlight alone. Stepping reuses the same `Range` objects, so
 * the match highlight stays registered and a step costs one range, not N.
 */
function publishActive(): void {
  const registry = highlightRegistry()
  const active = ranges[activeIndex]

  if (!registry) {
    return
  }

  // Re-insert (not overwrite): `CSS.highlights` is a Map, so an existing key
  // keeps its original paint position and the active match could land under
  // the all-matches highlight.
  registry.delete(ACTIVE_HIGHLIGHT)

  if (active) {
    const only = new Highlight()

    only.add(active)
    registry.set(ACTIVE_HIGHLIGHT, only)
  }
}

/** Drop every painted match. Module state, not the DOM — nothing to unwrap. */
function dropHighlights(): void {
  const registry = highlightRegistry()

  registry?.delete(ALL_HIGHLIGHT)
  registry?.delete(ACTIVE_HIGHLIGHT)
  ranges = []
  activeIndex = 0
}

/** Scroll the active match into view. Block: 'nearest' so a match already on
 *  screen doesn't twitch; guarded because jsdom has no scrollIntoView and the
 *  bar must still work where the renderer side has no layout. */
function scrollActiveIntoView(): void {
  const active = ranges[activeIndex]
  const element = active?.startContainer.parentElement ?? null

  if (element && typeof element.scrollIntoView === 'function') {
    element.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }
}

/** Result of a find / step — the same shape the bar already shows. */
export interface ScopedFindResult {
  count: number
  activeOrdinal: number
}

export interface ScopedFindOptions {
  forward: boolean
  findNext: boolean
}

const DEFAULT_RESULT: ScopedFindResult = { count: 0, activeOrdinal: 0 }

/**
 * Run a scoped find against `root`. When `findNext` is true and the query is
 * unchanged on a clean scope, advance / step the active match without
 * re-scanning; otherwise collect the matches for `query` and start at the
 * first one (or the last, entering backwards).
 *
 * Returns the (count, activeOrdinal) the bar should display. Returning a
 * plain object instead of pushing into the store keeps this helper testable
 * without a nanostores harness — the store wires it up.
 */
export function performScopedFind(root: Element, query: string, options: ScopedFindOptions): ScopedFindResult {
  if (!query) {
    activeQuery = ''
    dirty = false
    dropHighlights()
    stopObserver()

    return DEFAULT_RESULT
  }

  // Step only when we hold ranges for THIS query and nothing has invalidated
  // them. A dirty scope means a render may have detached them (or added
  // matches), so re-scan first and land on a live match rather than a stale
  // ordinal (the old node-wrapping engine needed the same guard, #81778).
  const stepping = Boolean(options.findNext) && activeQuery === query && ranges.length > 0 && !dirty

  if (stepping) {
    activeIndex = options.forward
      ? (activeIndex + 1) % ranges.length
      : (activeIndex - 1 + ranges.length) % ranges.length
    publishActive()
    scrollActiveIntoView()

    return { count: ranges.length, activeOrdinal: activeIndex + 1 }
  }

  ranges = collectRanges(root, query)
  activeQuery = query
  lastScanAt = Date.now()
  dirty = false

  // Entering find mode backwards (Shift+Enter on the first press) lands on
  // the last match, matching browser convention.
  activeIndex = options.forward ? 0 : Math.max(0, ranges.length - 1)

  if (ranges.length === 0) {
    // A query that matches nothing has nothing to maintain.
    dropHighlights()
    stopObserver()

    return DEFAULT_RESULT
  }

  publishRanges()
  ensureObserver(root)
  scrollActiveIntoView()

  return { count: ranges.length, activeOrdinal: activeIndex + 1 }
}

// ── Re-scan on React re-render ──────────────────────────────────────────────
/**
 * Attach the scope watcher, if it isn't already attached. Only observes the
 * current scope; a fresh `captureFindScope` (or close) detaches it.
 */
function ensureObserver(root: Element): void {
  if (observer && scopeRoot === root) {
    return
  }

  stopObserver()
  observer = new MutationObserver(() => {
    // Ranges survive a re-render that leaves their text nodes attached and go
    // dark when it doesn't; either way the match set may have changed, so a
    // mutation only marks the scan dirty. The scan itself is throttled.
    dirty = true
    scheduleRescan()
  })
  observer.observe(root, { characterData: true, childList: true, subtree: true })
}

function stopObserver(): void {
  observer?.disconnect()
  observer = null

  if (rescanTimer !== null) {
    clearTimeout(rescanTimer)
    rescanTimer = null
  }
}

function scheduleRescan(): void {
  if (rescanTimer !== null || !scopeRoot || !activeQuery) {
    return
  }

  const wait = Math.max(0, RESCAN_MIN_INTERVAL_MS - (Date.now() - lastScanAt))

  rescanTimer = setTimeout(() => {
    rescanTimer = null
    rescanNow()
  }, wait)
}

/** Re-collect the matches for the active query after the scope changed.
 *  Deliberately does NOT scroll: this runs because a background render
 *  happened, and the reader's viewport is theirs. */
function rescanNow(): void {
  const root = scopeRoot

  if (!root || !activeQuery) {
    return
  }

  ranges = collectRanges(root, activeQuery)
  activeIndex = Math.min(activeIndex, Math.max(0, ranges.length - 1))
  lastScanAt = Date.now()
  dirty = false
  publishRanges()
}

/** Forget everything about the current search. Called when the bar re-opens or
 *  closes so state never leaks across searches. */
function resetFindState(): void {
  stopObserver()
  dropHighlights()
  scopeRoot = null
  activeQuery = ''
  dirty = false
  lastScanAt = 0
}

/** Tear down highlights and the scope marker — called when the bar closes. */
export function releaseFindScope(): void {
  resetFindState()
  document.querySelectorAll<HTMLElement>(`[${ROOT_ATTR}]`).forEach(root => root.removeAttribute(ROOT_ATTR))
}
