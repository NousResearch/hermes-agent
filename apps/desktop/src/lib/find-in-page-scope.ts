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
 * remember it for the lifetime of the find bar. Subsequent queries highlight
 * only text nodes inside that subtree. ⌘G / ⌘⇧G step between highlights
 * inside the same subtree. Closing the bar unwraps the highlights.
 *
 * "Current view" is the foreground `[data-chat-surface]` element after
 * filtering out inactive keep-alive tabs. That is the same policy every
 * other document-wide lookup obeys (see pane-visibility.ts), so this can be
 * reasoned about as "find wherever the user is reading".
 *
 * Implementation notes:
 * - We wrap each match in a `<mark class="find-hit">` so the active match is
 *   a single class toggle. Walking raw text nodes lets us split a node that
 *   spans a match boundary without disturbing the surrounding React tree
 *   (text nodes are inert for React reconciliation until the parent changes).
 * - The active ordinal is the 1-indexed position of the match whose
 *   `<mark>` carries `data-active`. We re-derive it on every step instead of
 *   keeping a counter, so a stale count after a DOM mutation self-heals on
 *   the next search.
 * - We do NOT call `webContents.findInPage` from here. That bridge still
 *   exists for multi-window secondary sessions (each window searches its
 *   own webContents), but for the primary window it cannot be scoped, so
 *   the renderer-side walker is the only path that satisfies the issue.
 */

import { queryVisible } from '@/components/pane-shell/pane-visibility'

const SCOPE_SELECTOR = '[data-chat-surface]'
const HIGHLIGHT_CLASS = 'find-hit'
const ACTIVE_ATTR = 'data-find-active'
const ROOT_ATTR = 'data-find-root'

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

  if (root) {
    root.setAttribute(ROOT_ATTR, '')
  }

  return root
}

/** The scope captured by the open bar, or null if none is active. */
export function currentFindScope(): HTMLElement | null {
  const roots = document.querySelectorAll<HTMLElement>(`[${ROOT_ATTR}]`)

  for (const root of roots) {
    if (!isElementInHiddenPane(root)) {
      return root
    }
  }

  return null
}

/** Same predicate pane-visibility exposes, kept local so this module is
 *  independently testable without an import cycle in the renderer. */
function isElementInHiddenPane(element: Element): boolean {
  return Boolean(element.closest('[data-pane-hidden]'))
}

/**
 * Walk `root`'s text nodes in document order, wrapping every (case-
 * insensitive) occurrence of `query` in a `<mark class="find-hit">`.
 *
 * Splits text nodes that span a match boundary so React (or any other
 * framework holding the parent) sees the text-node change as a normal DOM
 * mutation. Returns the marks in document order — the same order the step
 * functions use to walk forward / backward.
 *
 * Implementation: walk forward, accumulating the global text offset and the
 * active text node as we go. When we hit a match, split the text node,
 * wrap the matched slice, and JUMP the offset past the wrapped region so we
 * don't re-match inside our own `<mark>`. Crucially, we update the text-
 * node pointer AFTER the wrap, so the next search step sees the new node
 * layout rather than the (now detached) original.
 */
function highlightMatches(root: Element, query: string): HTMLElement[] {
  const marks: HTMLElement[] = []
  const lowerQuery = query.toLowerCase()

  if (!lowerQuery) {
    return marks
  }

  let current: Node | null = root.firstChild
  let textNode: Text | null = null

  // Outer loop: walk element children until we have a text node to search.
  while (current) {
    if (current.nodeType === Node.TEXT_NODE) {
      textNode = current as Text
    } else if (current.nodeType === Node.ELEMENT_NODE) {
      const el = current as Element

      // Skip elements we never want to search into. The walker is
      // deliberately flat (no recursion); a nested <p> in a <section>
      // becomes a fresh descendant scan via the recursive call below.
      if (shouldSkipElement(el)) {
        current = el.nextSibling

        continue
      }

      // Recurse into the element's descendants.
      const inner = highlightMatches(el, query)

      marks.push(...inner)
      current = el.nextSibling

      continue
    } else {
      current = current.nextSibling

      continue
    }

    if (!textNode || !textNode.parentNode) {
      current = textNode?.nextSibling ?? null

      continue
    }

    // Search within this text node. Splits may invalidate `textNode`'s
    // identity, so we re-read it from the parent each iteration.
    let nodeValue = textNode.nodeValue ?? ''
    // Capture the text node's own next sibling BEFORE any replaceChild:
    // once the original node is detached, `.nextSibling` reads null and
    // the outer walker would stop, skipping every following sibling
    // subtree (`<div>needle<span>needle</span></div>` searching "needle"
    // matched only the first). `continueFrom` is the sibling AFTER this
    // text node — for a fully-consumed match, the walker resumes there.
    const continueFrom = textNode.nextSibling

    while (true) {
      const lower = nodeValue.toLowerCase()
      const idx = lower.indexOf(lowerQuery)

      if (idx === -1) {
        break
      }

      const before = nodeValue.slice(0, idx)
      const matchText = nodeValue.slice(idx, idx + lowerQuery.length)
      const after = nodeValue.slice(idx + lowerQuery.length)
      const parent = textNode.parentNode

      if (!parent) {
        break
      }

      const fragment = document.createDocumentFragment()

      if (before) {
        fragment.appendChild(document.createTextNode(before))
      }

      const mark = document.createElement('mark')

      mark.className = HIGHLIGHT_CLASS
      mark.textContent = matchText
      fragment.appendChild(mark)

      // Capture the after-sibling BEFORE replaceChild — replaceChild moves
      // the fragment's children into the parent and empties the fragment,
      // so looking at `fragment.lastChild` afterwards would point at a
      // detached `<mark>` (or null if `before` was empty).
      const afterNode = after ? document.createTextNode(after) : null

      if (afterNode) {
        fragment.appendChild(afterNode)
      }

      parent.replaceChild(fragment, textNode)

      marks.push(mark)

      if (!afterNode) {
        // Whole text node consumed — nothing left to scan in this region.
        textNode = null

        break
      }

      textNode = afterNode
      nodeValue = afterNode.nodeValue ?? ''
    }

    if (textNode) {
      current = textNode.nextSibling
    } else {
      // The whole text node was consumed by matches — `textNode` was
      // detached by replaceChild so its own `.nextSibling` is null. Resume
      // the outer walker from the parent's next sibling (captured before
      // the first replaceChild), so the sibling subtree is still searched.
      current = continueFrom
    }
  }

  return marks
}

/** Elements we never want to descend into during a search. Mirrors the
 *  filter the TreeWalker applied in the original design. */
function shouldSkipElement(el: Element): boolean {
  const tag = el.tagName

  if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT') {
    return true
  }

  if (el.closest(`mark.${HIGHLIGHT_CLASS}`)) {
    return true
  }

  if (el.closest('[role="search"]')) {
    return true
  }

  return false
}

/**
 * True when `root` contains an occurrence of `query` that is NOT inside one
 * of our find-hit marks. Read-only counterpart of `highlightMatches` with
 * the identical skip rules, used by the findNext fast path to verify that
 * the existing marks still cover every match — a stale mark from an earlier
 * query can satisfy the per-mark text comparison yet leave live occurrences
 * unwrapped, and stepping on that set would never highlight them.
 */
function hasUnmarkedMatch(root: Element, query: string): boolean {
  const lowerQuery = query.toLowerCase()

  if (!lowerQuery) {
    return false
  }

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)

  while (walker.nextNode()) {
    const node = walker.currentNode as Text
    const parent = node.parentElement

    if (parent && shouldSkipElement(parent)) {
      continue
    }

    if (node.nodeValue?.toLowerCase().includes(lowerQuery)) {
      return true
    }
  }

  return false
}

/** Remove every find-hit mark we previously added, restoring original text. */
function clearHighlights(root: Element): void {
  const marks = root.querySelectorAll<HTMLElement>(`mark.${HIGHLIGHT_CLASS}`)

  for (const mark of marks) {
    const parent = mark.parentNode

    if (!parent) {
      continue
    }

    while (mark.firstChild) {
      parent.insertBefore(mark.firstChild, mark)
    }

    parent.removeChild(mark)
    parent.normalize()
  }
}

/** Mark one element as the active match and scroll it into view. */
function setActiveMark(mark: HTMLElement | null): void {
  document.querySelectorAll(`mark[${ACTIVE_ATTR}]`).forEach(el => el.removeAttribute(ACTIVE_ATTR))

  if (!mark) {
    return
  }

  mark.setAttribute(ACTIVE_ATTR, '')
  // Block: 'nearest' so a match already on screen doesn't twitch, but a
  // match below the fold scrolls into view instead of silently landing off-
  // screen. Inline: 'nearest' for the same reason. Guarded: jsdom does not
  // implement scrollIntoView, and the bar still needs to highlight even
  // when the renderer side has no layout (tests, headless boot, …).
  if (typeof mark.scrollIntoView === 'function') {
    mark.scrollIntoView({ block: 'nearest', inline: 'nearest' })
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
 * Run a scoped find against `root`. When `findNext` is true, advance / step
 * the active mark without re-highlighting (the query has not changed). When
 * false, drop prior highlights and re-wrap matches for `query`.
 *
 * Returns the (count, activeOrdinal) the bar should display. Returning a
 * plain object instead of pushing into the store keeps this helper testable
 * without a nanostores harness — the store wires it up.
 */
export function performScopedFind(
  root: Element,
  query: string,
  options: ScopedFindOptions
): ScopedFindResult {
  if (!query) {
    clearHighlights(root)
    setActiveMark(null)

    return DEFAULT_RESULT
  }

  const existingMarks = [...root.querySelectorAll<HTMLElement>(`mark.${HIGHLIGHT_CLASS}`)]
  // The marks store the ORIGINAL-CASE source slice (`highlightMatches`
  // writes `mark.textContent = matchText`), so byte-equality against the
  // typed query fails on the first match whose casing differs — 'Hermes'
  // for 'hermes', sentence-initial capitals, ALL-CAPS. Without the
  // case-insensitive comparison, every Enter/⌘G step re-wraps all
  // highlights, `data-find-active` is lost on the fresh DOM, and the
  // active ordinal resets to 1 forever (triage finding on #81778).
  // The per-mark comparison alone is not enough: a stale mark from an
  // earlier query that survived a raced re-wrap (or external DOM writes)
  // can match the current query case-insensitively while the mark set no
  // longer covers every match — stepping would walk old highlights and
  // never wrap the live ones. Only step when the marks match AND cover
  // every occurrence (review finding on #81778).
  const sameQuery =
    options.findNext &&
    existingMarks.length > 0 &&
    existingMarks.every(mark => mark.textContent.toLowerCase() === query.toLowerCase()) &&
    !hasUnmarkedMatch(root, query)

  let marks = existingMarks

  if (!sameQuery) {
    clearHighlights(root)
    marks = highlightMatches(root, query)
  }

  if (marks.length === 0) {
    setActiveMark(null)

    return DEFAULT_RESULT
  }

  const previousActive = root.querySelector<HTMLElement>(`mark[${ACTIVE_ATTR}]`)
  let nextIndex = 0

  if (previousActive) {
    const previousIndex = marks.indexOf(previousActive)

    if (previousIndex !== -1) {
      nextIndex = options.forward
        ? (previousIndex + 1) % marks.length
        : (previousIndex - 1 + marks.length) % marks.length
    }
  } else if (!options.forward) {
    // Entering find mode backwards (Shift+Enter on first press): land on the
    // last match, matching browser convention.
    nextIndex = marks.length - 1
  }

  setActiveMark(marks[nextIndex] ?? null)

  return { count: marks.length, activeOrdinal: nextIndex + 1 }
}

/** Tear down highlights and the scope marker — called when the bar closes. */
export function releaseFindScope(): void {
  const roots = document.querySelectorAll<HTMLElement>(`[${ROOT_ATTR}]`)

  for (const root of roots) {
    clearHighlights(root)
    root.removeAttribute(ROOT_ATTR)
  }

  setActiveMark(null)
}