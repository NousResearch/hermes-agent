import type { ChatMessage } from '@/lib/chat-messages'
import { messageRenderWeight } from '@/lib/render-weight'

/**
 * Bound what reaches assistant-ui at all.
 *
 * Rendering the full transcript of an oversized session rebuilds an unbounded
 * runtime repository on every store update and exhausts the renderer's V8 heap
 * (#55191). The DOM budget in `thread/list.tsx` already bounds what PAINTS, but
 * every message still gets normalized into the repository first — so a session
 * only has to be heavy, not visible, to crash the window.
 *
 * The window spends the same currency as the DOM budget: render weight, not
 * message count. A count cap gets this wrong in both directions — measured on a
 * real 1,175-session store, a 400-message cap would disable itself on 37
 * sessions that are heavy but short (one was 133 messages / 1.05MB) while
 * engaging on 92 long-but-light sessions that were never at risk.
 */

/**
 * One window page, in render-weight units.
 *
 * Four DOM pages (the `RENDER_BUDGET` of 300 in `thread/list.tsx`). "Show
 * earlier" spends the DOM budget first, so the user pages through the
 * already-materialized window three times before this asks the store for more
 * — and the reported crash shape (~231K tokens ≈ 2,260 units) is windowed
 * rather than handed to the repository whole.
 */
export const TRANSCRIPT_WINDOW_BUDGET = 1200

/**
 * Floor on messages kept regardless of weight. A transcript of enormous turns
 * must still render the turn the user is having; without this a single
 * multi-megabyte tool result could window everything after it away.
 */
export const TRANSCRIPT_WINDOW_MIN_MESSAGES = 30

export interface TranscriptWindow {
  /** The tail assistant-ui is allowed to materialize. */
  messages: ChatMessage[]
  /** Store holds older messages than this window. */
  windowed: boolean
}

/**
 * Widen a cut backwards so it never lands inside an assistant branch group.
 *
 * `useRuntimeMessageRepository` records a group's fork point the first time it
 * sees the group (`branchParentByGroup`). A cut through the middle of a group
 * therefore anchors the surviving branches to whatever message happens to
 * precede them in the window — silently re-parenting a branch. Include the
 * whole group or none of it.
 */
export function alignToBranchGroup(messages: readonly ChatMessage[], start: number): number {
  if (start <= 0 || start >= messages.length) {
    return Math.max(0, Math.min(start, messages.length))
  }

  const group = messages[start].branchGroupId

  if (!group) {
    return start
  }

  let aligned = start

  while (aligned > 0 && messages[aligned - 1].branchGroupId === group) {
    aligned--
  }

  return aligned
}

/**
 * Select the tail of the transcript that fits one window, grown by `pages`.
 *
 * Walks newest-first accumulating weight until the budget is met, keeps at
 * least MIN messages, then aligns the cut off a branch-group boundary.
 */
export function selectTranscriptWindow(messages: readonly ChatMessage[], pages = 1): TranscriptWindow {
  const budget = TRANSCRIPT_WINDOW_BUDGET * Math.max(1, Math.floor(pages))

  if (messages.length === 0) {
    return { messages: messages as ChatMessage[], windowed: false }
  }

  let start = messages.length
  let weight = 0

  for (let i = messages.length - 1; i >= 0; i--) {
    weight += messageRenderWeight(messages[i].parts)
    start = i

    if (weight >= budget && messages.length - i >= TRANSCRIPT_WINDOW_MIN_MESSAGES) {
      break
    }
  }

  start = alignToBranchGroup(messages, start)

  if (start <= 0) {
    // Preserve reference identity when the whole transcript fits: handing React
    // a fresh array of the same messages re-renders the runtime for nothing.
    return { messages: messages as ChatMessage[], windowed: false }
  }

  return { messages: messages.slice(start), windowed: true }
}
