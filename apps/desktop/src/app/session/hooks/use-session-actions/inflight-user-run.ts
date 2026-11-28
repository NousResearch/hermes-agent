import type { ChatMessage } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

/** User rows belonging to the latest turn, across its live or persisted tool rounds. */
export function latestInflightUserRun(
  messages: ChatMessage[],
  latestUserIndex: number,
  turnStartedAt: number | null | undefined
): ChatMessage[] {
  const users: ChatMessage[] = []

  for (let index = latestUserIndex; index >= 0; index -= 1) {
    const candidate = messages[index]

    if (candidate.role === 'user') {
      users.unshift(candidate)

      continue
    }

    // Tool rounds commit before the final answer. Their durable rows still belong
    // to this turn. A previous turn can finish on a tool-bearing row, so only
    // cross a durable partial when its timestamp proves it belongs to this turn.
    const currentToolRound =
      candidate.durableComplete === false &&
      typeof turnStartedAt === 'number' &&
      typeof candidate.timestamp === 'number' &&
      candidate.timestamp >= turnStartedAt

    if (
      candidate.role === 'assistant' &&
      (candidate.pending === true ||
        isLiveTailReplyId(candidate.id) ||
        candidate.interim === true ||
        currentToolRound)
    ) {
      continue
    }

    break
  }

  return users
}
