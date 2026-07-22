const MESSAGE_SELECTOR = '[data-message-id]'

/**
 * Resolve the active timeline prompt from the bounded set of message nodes that
 * is actually rendered. The transcript can contain thousands of entries while
 * ThreadMessageList keeps only a small part budget in the DOM, so querying once
 * and filtering through the timeline index avoids one selector scan per turn.
 */
export function activeTimelineIndexInViewport(
  viewport: HTMLElement,
  entryIndexById: ReadonlyMap<string, number>,
  slack = 8
): number {
  const rendered: { index: number; node: HTMLElement }[] = []

  for (const node of viewport.querySelectorAll<HTMLElement>(MESSAGE_SELECTOR)) {
    const id = node.dataset.messageId
    const index = id ? entryIndexById.get(id) : undefined

    if (index !== undefined) {
      rendered.push({ index, node })
    }
  }

  if (rendered.length === 0) {
    return 0
  }

  // Message roots follow transcript order, and each turn bounds its sticky user
  // root, so their vertical positions remain monotonic. Binary-search the last
  // prompt at/above the viewport edge instead of forcing layout for every root.
  const edge = viewport.getBoundingClientRect().top + slack
  let low = 0
  let high = rendered.length - 1
  let active = -1

  while (low <= high) {
    const middle = Math.floor((low + high) / 2)

    if (rendered[middle].node.getBoundingClientRect().top <= edge) {
      active = middle
      low = middle + 1
    } else {
      high = middle - 1
    }
  }

  return active === -1 ? rendered[0].index : rendered[active].index
}
