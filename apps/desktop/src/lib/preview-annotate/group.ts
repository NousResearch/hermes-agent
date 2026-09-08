/**
 * Structural grouping for a comment batch.
 *
 * Twenty-three comments used to arrive as twenty-three flat blocks, so the
 * agent made twenty-three todos and worked them one at a time. The fix is not
 * a classifier in the renderer — "is this a UI nit or a functional bug" is a
 * judgment only the model can make, and prose-matching it here would be wrong
 * constantly. What the renderer CAN know is structure: which pins sit in the
 * same part of the DOM, and therefore which ones are likely the same component
 * and the same source file.
 *
 * So this splits the batch by shared ancestor path and hands the model groups
 * that touch disjoint subtrees. Disjoint is the property that makes parallel
 * work safe — grouping by theme instead ("all the UI ones") would put five
 * agents in the same files. The model still owns the semantics and can regroup;
 * these are labelled starting points, not orders.
 *
 * Two properties keep the split honest without a tuning knob:
 *
 * - It compares ANCESTOR paths, not full selectors. Two comments on the heading
 *   and the paragraph of one card differ at the leaf, and splitting there would
 *   hand out singletons — the thing this exists to prevent. Their parents are
 *   identical, so they group.
 * - The depth is derived, then refined: descend the shared prefix until it
 *   stops being shared, and sub-split any group that ends up holding most of
 *   the batch. So the group count follows the page the user commented on rather
 *   than a constant someone picked.
 */

import type { ComposerReadyAnnotation } from './pack'

export interface AnnotateGroup {
  /** Shared ancestor prefix, or '' for the group that has no element. */
  key: string
  items: ComposerReadyAnnotation[]
  /** Short human label for the shared region, e.g. `section.hero`. */
  label: string
}

const SEP = '>'

function segments(selector: string): string[] {
  return selector.split(SEP).filter(Boolean)
}

/**
 * The element's container. A one-segment selector is its own container —
 * dropping to nothing would collide with the unanchored group's empty key.
 */
function ancestorPath(selector: string): string[] {
  const parts = segments(selector)

  return parts.length > 1 ? parts.slice(0, -1) : parts
}

function prefixAt(parts: string[], depth: number): string {
  return parts.slice(0, depth).join(SEP)
}

/**
 * First depth at which the ancestor paths stop agreeing.
 *
 * Grouping by a prefix of this depth yields the top-level regions the user
 * touched. When every path is identical there is no boundary and everything
 * belongs to one group.
 */
export function annotateSplitDepth(selectors: readonly string[]): number {
  const parts = selectors.map(ancestorPath)

  if (parts.length < 2) {
    return parts[0]?.length ? 1 : 0
  }

  const shortest = Math.min(...parts.map(list => list.length))

  for (let depth = 1; depth <= shortest; depth++) {
    const seen = new Set(parts.map(list => prefixAt(list, depth)))

    if (seen.size > 1) {
      return depth
    }
  }

  // Every path shares the whole of the shortest one: the shorter paths are
  // ancestors of the longer ones, so one segment deeper is where they part.
  return parts.some(list => list.length > shortest) ? shortest + 1 : shortest
}

function labelFor(key: string): string {
  const parts = segments(key)

  return parts[parts.length - 1] || ''
}

function bucket(items: readonly ComposerReadyAnnotation[], depth: number): AnnotateGroup[] {
  const byKey = new Map<string, AnnotateGroup>()

  for (const item of items) {
    const key = prefixAt(ancestorPath(item.identity?.selector || ''), depth)
    const group = byKey.get(key)

    if (group) {
      group.items.push(item)

      continue
    }

    byKey.set(key, { items: [item], key, label: labelFor(key) })
  }

  return Array.from(byKey.values())
}

/**
 * One pass of the split leaves the deepest branch lumped together: on a normal
 * page `header`, `main`, and `footer` part company at the top, so every comment
 * inside `main` — hero, pricing, faq — lands in one oversized group. That group
 * is not foldable into a single change and not safely divisible among workers,
 * which is the whole point of grouping.
 *
 * So refine: while some group holds more than a third of the batch and its
 * members do diverge further down, replace it with its own sub-split. A group
 * holding most of the batch has not separated anything. Each pass strictly
 * shrinks the largest group or finds it indivisible, so this terminates.
 */
function refine(groups: AnnotateGroup[], total: number): AnnotateGroup[] {
  const ceiling = Math.max(2, Math.ceil(total / 3))
  let current = groups

  for (let pass = 0; pass < total; pass++) {
    const target = current.find(group => group.items.length > ceiling)

    if (!target) {
      break
    }

    const selectors = target.items.map(item => item.identity?.selector || '')
    const deeper = annotateSplitDepth(selectors)
    const split = bucket(target.items, deeper)

    if (split.length < 2) {
      break
    }

    current = current.flatMap(group => (group === target ? split : [group]))
  }

  return current
}

/**
 * Split a packed batch into groups the model can hand out in parallel.
 *
 * Comments with no element (area pins) cannot be placed in the tree, so they
 * collect in one trailing group rather than being guessed into someone else's
 * subtree. Group order follows first appearance, so numbering still reads in
 * the order the user clicked.
 */
export function groupAnnotations(items: readonly ComposerReadyAnnotation[]): AnnotateGroup[] {
  const placed = items.filter(item => item.identity?.selector)
  const loose = items.filter(item => !item.identity?.selector)
  const depth = annotateSplitDepth(placed.map(item => item.identity?.selector || ''))
  const groups = refine(bucket(placed, depth), placed.length)

  if (loose.length) {
    groups.push({ items: [...loose], key: '', label: '' })
  }

  return groups
}
