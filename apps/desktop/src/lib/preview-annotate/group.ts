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
 * A comment on a container plus comments inside it must stay one group.
 * Nested keys would tell the model those pieces of work are safe to run in
 * parallel, and they are not.
 *
 * Walk the clicked selectors, not the bucket keys. A comment on `main` can
 * bucket as `body`, and that key is a prefix of `body>header.nav` even though
 * header is a sibling. The element the user actually pointed at is what
 * decides ownership. Empty-key unanchored comments never swallow a placed
 * group.
 */
function flattenNested(groups: AnnotateGroup[]): AnnotateGroup[] {
  const firstNumber = (group: AnnotateGroup) => group.items[0]?.number ?? 0
  const selectorsOf = (group: AnnotateGroup) =>
    group.items.map(item => item.identity?.selector || '').filter(Boolean)
  const owns = (parent: AnnotateGroup, child: AnnotateGroup): boolean => {
    const parents = selectorsOf(parent)
    const children = selectorsOf(child)

    if (!parents.length || !children.length) {
      return false
    }

    return children.every(childSel =>
      parents.some(parentSel => childSel === parentSel || childSel.startsWith(`${parentSel}>`))
    )
  }
  const sorted = [...groups].sort((left, right) => {
    const leftMin = Math.min(...selectorsOf(left).map(sel => sel.length))
    const rightMin = Math.min(...selectorsOf(right).map(sel => sel.length))

    if (leftMin !== rightMin) {
      return leftMin - rightMin
    }

    return firstNumber(left) - firstNumber(right)
  })
  const kept: AnnotateGroup[] = []

  for (const group of sorted) {
    const parent = kept.find(candidate => candidate.key !== '' && owns(candidate, group))

    if (parent) {
      parent.items.push(...group.items)
      parent.items.sort((left, right) => left.number - right.number)
      const key = sharedSelectorPrefix(selectorsOf(parent))
      parent.key = key
      parent.label = labelFor(key)
      continue
    }

    kept.push({ items: [...group.items], key: group.key, label: group.label })
  }

  kept.sort((left, right) => firstNumber(left) - firstNumber(right))

  return kept
}

function sharedSelectorPrefix(selectors: string[]): string {
  if (!selectors.length) {
    return ''
  }

  const parts = selectors.map(segments)
  const shortest = Math.min(...parts.map(list => list.length))
  const common: string[] = []

  for (let i = 0; i < shortest; i++) {
    const seg = parts[0]?.[i]

    if (!seg || parts.some(list => list[i] !== seg)) {
      break
    }

    common.push(seg)
  }

  return common.join(SEP)
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
  const groups = flattenNested(refine(bucket(placed, depth), placed.length))

  if (loose.length) {
    groups.push({ items: [...loose], key: '', label: '' })
  }

  return groups
}
