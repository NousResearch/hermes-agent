/**
 * Pure range logic for the work block (Group work between replies).
 *
 * assistant-ui's chain-of-thought mode hands the transcript one range per
 * stretch of reasoning + tool calls between two visible texts (blank texts
 * never reach it: fromThreadMessageLike drops them). The work block folds the
 * range behind one line, except for parts that must stay on screen (questions,
 * consent cards, subagent cards, the theme picker, images, delivery notices):
 * those split the range and render in place, with a new line after them.
 */

export interface WorkPartLike {
  /** A tool call that must stay on screen. */
  stays?: boolean
  type: string
}

export type WorkItem = { end: number; kind: 'block'; start: number } | { index: number; kind: 'part' }

/** Split `[start, end]` into foldable blocks and parts that stay visible. */
export function splitWorkRange(parts: readonly (WorkPartLike | undefined)[], start: number, end: number): WorkItem[] {
  const items: WorkItem[] = []
  let blockStart: null | number = null

  for (let index = start; index <= end; index++) {
    if (parts[index]?.type === 'tool-call' && parts[index]?.stays) {
      if (blockStart !== null) {
        items.push({ end: index - 1, kind: 'block', start: blockStart })
        blockStart = null
      }

      items.push({ index, kind: 'part' })
    } else if (blockStart === null) {
      blockStart = index
    }
  }

  if (blockStart !== null) {
    items.push({ end, kind: 'block', start: blockStart })
  }

  return items
}
