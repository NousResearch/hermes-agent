import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { getLatestSessionMessages, type ProfileScope } from '@/hermes'
import { type ChatMessage, preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { knownSessionOwner, ownerLookupSessionRows } from '@/store/session'
import type { SessionOwnerScope } from '@/store/session-request-router'

/** REST scope for a session owner. Undefined when the owner is unknown. */
export function profileScopeForSessionOwner(owner: SessionOwnerScope): ProfileScope {
  if (!owner) {
    return undefined
  }

  if (typeof owner === 'string') {
    return owner
  }

  return {
    connectionId: owner.connectionId,
    profile: owner.targetProfile ?? owner.profile
  }
}

/**
 * Transcript content a view actually authored. Backend-written notices
 * (`ChatMessage.systemNotice`) render on the timeline but belong to no view, so
 * counting them reports a second window that does not exist: an in-place model
 * switch alone refused every send with "this window was behind another view of
 * the same chat".
 */
function authoredMessageCount(messages: ChatMessage[]): number {
  return messages.reduce((count, message) => (message.systemNotice ? count : count + 1), 0)
}

/**
 * Latest persisted backend row the view carries. Retention only ever releases
 * the head (rows older than the window plus its budget — see
 * app/chat/transcript-retention.ts), so the last durable row is always the
 * live tail; unpersisted rows (optimistic prompts, live streams) sit past it.
 *
 * A folded assistant bubble spans several rows but carries one `rowId`: the
 * live settle keeps `rowId: message.rowId ?? final_assistant_row_id` (the
 * turn's last row) while hydration keeps the bubble's first text row. Both
 * paths bind the final row on the bubble's last text part (`sourceRowId`), so
 * the tip is the highest durable address in the bubble (#125975).
 */
function bubbleTip(message: ChatMessage): number | undefined {
  let tip = message.rowId

  for (const part of message.parts) {
    if (part.type === 'text' && typeof part.sourceRowId === 'number') {
      tip = tip === undefined ? part.sourceRowId : Math.max(tip, part.sourceRowId)
    }
  }

  return tip
}

function lastDurableRowId(messages: readonly ChatMessage[]): number | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const tip = bubbleTip(messages[index])

    if (tip !== undefined) {
      return tip
    }
  }

  return undefined
}

/**
 * Chat messages to install when the authoritative latest page is ahead of the
 * local view. Null when the local view is current.
 *
 * Tips are compared before counts: a view ending on the page's own last
 * durable row is current however much head retention has paged out — counts
 * never converge there (#123909), while a peer window's newer row still
 * changes the tip. Authored content is compared after `toChatMessages`, so
 * tool rows folded into an assistant bubble are not "ahead", and neither is a
 * backend-authored notice. A backfilled prefix is kept when the refreshed tail
 * anchors inside it. Live stream ids that do not anchor still use the count,
 * so the window that just finished the turn is not blocked when the counts
 * match.
 */
export function messagesIfTranscriptBehind(
  localMessages: ChatMessage[],
  remoteChat: ChatMessage[]
): ChatMessage[] | null {
  if (remoteChat.length === 0) {
    return null
  }

  if (localMessages.length === 0) {
    return remoteChat
  }

  const localTip = lastDurableRowId(localMessages)
  const remoteTip = lastDurableRowId(remoteChat)

  if (localTip !== undefined && localTip === remoteTip) {
    return null
  }

  const grafted = graftRefreshedTailOntoBackfill(remoteChat, localMessages)
  const localAuthored = authoredMessageCount(localMessages)

  if (grafted === remoteChat) {
    return authoredMessageCount(remoteChat) > localAuthored ? remoteChat : null
  }

  return authoredMessageCount(grafted) > localAuthored ? grafted : null
}

/**
 * Read the authoritative latest page and return a refreshed transcript when
 * this view is behind. Null when current or the read fails — a missing
 * profile or a down backend must not soft-lock send.
 */
export async function refreshIfTranscriptStale(
  storedSessionId: string,
  localMessages: ChatMessage[],
  options?: { excludeMessageId?: string; profile?: ProfileScope }
): Promise<ChatMessage[] | null> {
  const baseline = options?.excludeMessageId
    ? localMessages.filter(message => message.id !== options.excludeMessageId)
    : localMessages

  const profile =
    options && 'profile' in options
      ? options.profile
      : profileScopeForSessionOwner(knownSessionOwner(ownerLookupSessionRows(), storedSessionId))

  try {
    const remote = await getLatestSessionMessages(storedSessionId, profile)
    const refreshed = messagesIfTranscriptBehind(baseline, toChatMessages(remote.messages))

    if (!refreshed) {
      return null
    }

    return preserveLocalAssistantErrors(refreshed, baseline)
  } catch {
    return null
  }
}
