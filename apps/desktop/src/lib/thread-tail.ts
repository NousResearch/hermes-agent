/**
 * THREAD TAIL — the foot of the transcript as a contribution area.
 *
 * `chat.empty` lets a plugin own a session's blank transcript. It has no
 * counterpart once messages exist: nothing a plugin registers can sit AFTER
 * the last message and stay there. Anything with a life of its own in the
 * conversation — a presence indicator that appears while the user is still
 * typing, a "someone is reading" row that lingers after the reply landed, a
 * per-session hint — has nowhere to live, because every other hook is scoped
 * to one message's lifecycle and `loadingIndicator` only renders inside the
 * populated branch.
 *
 * This area renders after BOTH branches (empty and populated) and reserves
 * composer clearance for itself, so a contribution is visible above the
 * floating composer whether the transcript has zero messages or a thousand.
 * With no contributions registered the transcript lays out exactly as before:
 * the slot costs nothing until a plugin claims it.
 *
 * Every registration is mounted and answers for itself (return `null` to
 * decline a session), mirroring `chat.empty`: ownership is per session and is
 * only known once each plugin has loaded its own data.
 */

import type { ReactNode } from 'react'

export const THREAD_TAIL_AREA = 'thread.tail'

/** Props handed to a thread-tail contribution's `render`. */
export interface ThreadTailProps {
  /** The live session whose transcript this tail belongs to. */
  sessionId: string
}

/** Payload of a `thread.tail` contribution's `data`. */
export interface ThreadTailContribution {
  /** Renders the tail content, or returns `null` to stand down for this
   *  session. Mounted as a component inside the contribution error boundary,
   *  so it may subscribe to its own stores; a throw degrades to an inline
   *  error, not a dead transcript. */
  render: (props: ThreadTailProps) => ReactNode
}
