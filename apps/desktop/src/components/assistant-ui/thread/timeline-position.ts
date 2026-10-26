const MESSAGE = '[data-message-id]'

interface MountedPrompt {
  element: HTMLElement
  index: number
}

/**
 * The transcript's outer groups are in normal-flow DOM order. Binary-search
 * their tops, not sticky bubbles or descendants of content-visibility boxes:
 * reading those descendants would force skipped markdown/tool trees to lay out.
 * Cache only membership, never geometry, so image loads and collapsing tools
 * cannot leave a stale offset table behind (#118782, #125766).
 *
 * `invalidate` sees every subtree mutation batch but never walks one: deep
 * records (streamed markdown, tool output) resolve with O(1) per-node checks,
 * and only the transcript root's direct children — where rows mount — pay a
 * containment scan. The full message sweep reruns solely when a rail-visible
 * message actually mounted or unmounted, not per streamed flush.
 */
export function createTimelinePositionReader(
  viewport: HTMLElement,
  indexes: ReadonlyMap<string, number>,
  content: HTMLElement = viewport
) {
  let prompts: MountedPrompt[] | null = null

  // A rail entry appears with its row at the transcript root or toggles its
  // data-message-id in place; anything deeper cannot change which prompts
  // exist. Attribute records must also accept the OLD value: removing the
  // attribute unmatches the node, and that removal has to drop the cached
  // prompt (attributeOldValue is enabled on the observer).
  const affectsMembership = (record: MutationRecord): boolean => {
    if (record.type === 'attributes') {
      const target = record.target

      return (
        target instanceof Element &&
        (indexes.has(target.getAttribute('data-message-id') ?? '') ||
          (record.oldValue !== null && indexes.has(record.oldValue)))
      )
    }

    const atRoot = record.target === content || record.target === viewport

    for (const node of [...record.addedNodes, ...record.removedNodes]) {
      if (!(node instanceof Element)) {
        continue
      }

      if (node.matches(MESSAGE)) {
        // Assistant bubbles stream in deep with ids the rail never lists;
        // only a rail-visible id can move the active tick.
        if (atRoot || indexes.has(node.getAttribute('data-message-id') ?? '')) {
          return true
        }
      } else if (atRoot && node.querySelector(MESSAGE) !== null) {
        return true
      }
    }

    return false
  }

  return {
    invalidate(records: readonly MutationRecord[]) {
      if (records.some(affectsMembership)) {
        prompts = null
      }
    },
    read(leadingIndex = -1): number {
      if (prompts === null) {
        prompts = []

        for (const node of viewport.querySelectorAll<HTMLElement>(MESSAGE)) {
          const index = indexes.get(node.dataset.messageId!)

          if (index !== undefined) {
            prompts.push({
              index,
              element:
                node.closest<HTMLElement>('[data-slot="aui_message-group"]') ??
                node.closest<HTMLElement>('[data-slot="aui_turn-pair"]') ??
                node
            })
          }
        }
      }

      if (!prompts.length) {
        return Math.max(0, leadingIndex)
      }

      const top = viewport.getBoundingClientRect().top + 8
      let before = 0
      let after = prompts.length

      while (before < after) {
        const middle = Math.floor((before + after) / 2)

        if (prompts[middle].element.getBoundingClientRect().top <= top) {
          before = middle + 1
        } else {
          after = middle
        }
      }

      return before > 0 ? prompts[before - 1].index : leadingIndex >= 0 ? leadingIndex : prompts[0].index
    }
  }
}
