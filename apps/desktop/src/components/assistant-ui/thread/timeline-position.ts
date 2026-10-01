const MESSAGE = '[data-message-id]'

interface MountedPrompt {
  element: HTMLElement
  index: number
}

const containsMessage = (node: Node): boolean =>
  node instanceof Element && (node.matches(MESSAGE) || node.querySelector(MESSAGE) !== null)

/**
 * The transcript's outer groups are in normal-flow DOM order. Binary-search
 * their tops, not sticky bubbles or descendants of content-visibility boxes:
 * reading those descendants would force skipped markdown/tool trees to lay out.
 * Cache only membership, never geometry, so image loads and collapsing tools
 * cannot leave a stale offset table behind (#118782, #125766).
 */
export function createTimelinePositionReader(viewport: HTMLElement, indexes: ReadonlyMap<string, number>) {
  let prompts: MountedPrompt[] | null = null

  return {
    invalidate(records: readonly MutationRecord[]) {
      if (
        records.some(
          record =>
            record.type === 'attributes' ||
            [...record.addedNodes, ...record.removedNodes].some(containsMessage)
        )
      ) {
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
