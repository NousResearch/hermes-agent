import type { ReactNode } from 'react'

/**
 * CHAT EMPTY STATE — the blank transcript as a contribution area.
 *
 * Core owns exactly one empty state: the intro splash, which belongs to a
 * fresh draft with no session selected. A session that exists but has nothing
 * in it yet falls outside that — and whoever opened it is the only one who
 * knows what should stand in the gap. A bot's chat wants the bot's face and
 * name; core has neither, and should not learn them.
 *
 * So the slot is contributed. A plugin claims the sessions it owns by
 * returning an element for them and `null` for everything else, which is also
 * how it stands down while the transcript is still hydrating.
 */

export const CHAT_EMPTY_AREA = 'chat.empty'

/** Props handed to a chat-empty contribution's `render`. */
export interface ChatEmptyProps {
  /** The live session whose transcript is empty. */
  sessionId: string
}

/** Payload of a `chat.empty` contribution's `data`. */
export interface ChatEmptyContribution {
  /** Renders the empty state, or returns `null` to decline the session and
   *  leave the transcript blank. Mounted as a component inside the
   *  contribution error boundary, so it can subscribe to its own stores and
   *  appear once they load — the roster a bot chat needs arrives after the
   *  transcript does. A throw degrades to an inline error, not a dead chat. */
  render: (props: ChatEmptyProps) => ReactNode
}
