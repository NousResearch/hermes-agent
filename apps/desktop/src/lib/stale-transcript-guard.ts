import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { getLatestSessionMessages, type ProfileScope } from '@/hermes'
import { type ChatMessage, chatMessageText, preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
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

export interface TranscriptRefresh {
  messages: ChatMessage[]
  /**
   * The refreshed surplus carries a user row this window never streamed: another
   * view of the same chat is ahead, and the pre-send guard must refuse (fork
   * risk). False when the surplus is this window's own server-side residue — a
   * turn that died on an approval timeout leaves its tool/assistant rows
   * server-side while the window only holds its optimistic user message, whose
   * id/rowId never reconciled. Graft silently and let the send proceed (#124005).
   */
  competingView: boolean
}

/**
 * Whether the refreshed transcript carries a user row this window does not know.
 * Durable ids and row ids are authoritative. The only text match allowed is the
 * current optimistic prompt identified by `optimisticMessageId`; matching all
 * user text would mistake a repeated prompt from another window for this
 * window's own turn. Assistant/tool-only surplus rows are this window's own
 * turn residue (#124005).
 */
export function surplusIsCompetingView(
  localMessages: ChatMessage[],
  refreshed: ChatMessage[],
  options?: { optimisticMessageId?: string }
): boolean {
  const localUsers = localMessages.filter(message => message.role === 'user')
  const matchedLocalUsers = new Set<number>()

  return refreshed.some(message => {
    if (message.role !== 'user') {
      return false
    }

    const matchIndex = localUsers.findIndex(
      (local, index) =>
        !matchedLocalUsers.has(index) &&
        (local.id === message.id ||
          (message.rowId !== undefined && local.rowId === message.rowId) ||
          (local.id === options?.optimisticMessageId && local.rowId === undefined &&
            chatMessageText(local) === chatMessageText(message)))
    )

    if (matchIndex < 0) {
      return true
    }

    matchedLocalUsers.add(matchIndex)
    return false
  })
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
 * Highest durable row address a bubble carries: its own `rowId` plus any text
 * part's `sourceRowId`. A folded tool-turn bubble spans many stored rows, and
 * the two paths that build it bind different ends — live settle stamps the
 * turn's final row (use-message-stream's withPersistedIdentity) while
 * hydration keeps the folded bubble's first row — so the tip must read every
 * address the bubble owns (#125975).
 */
function bubbleTipRowId(message: ChatMessage): number | undefined {
  let tip = message.rowId

  for (const part of message.parts) {
    if (part.type === 'text' && typeof part.sourceRowId === 'number') {
      tip = tip === undefined ? part.sourceRowId : Math.max(tip, part.sourceRowId)
    }
  }

  return tip
}

/**
 * Latest persisted backend row the view carries. Retention only ever releases
 * the head (rows older than the window plus its budget — see
 * app/chat/transcript-retention.ts), so the last durable row is always the
 * live tail; unpersisted rows (optimistic prompts, live streams) sit past it.
 */
function lastDurableRowId(messages: readonly ChatMessage[]): number | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const rowId = bubbleTipRowId(messages[index])

    if (typeof rowId === 'number') {
      return rowId
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

async function fetchRefreshedTranscript(
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
  return fetchRefreshedTranscript(storedSessionId, localMessages, options)
}

/**
 * Same read, plus whether the surplus is a competing view's message. The
 * pre-send guard refuses only a competing view; this window's own server-side
 * turn residue (a turn that died on an approval timeout) is grafted silently
 * and the send proceeds (#124005).
 */
export async function transcriptRefreshIfBehind(
  storedSessionId: string,
  localMessages: ChatMessage[],
  options?: { excludeMessageId?: string; profile?: ProfileScope }
): Promise<TranscriptRefresh | null> {
  const messages = await fetchRefreshedTranscript(storedSessionId, localMessages, options)

  if (!messages) {
    return null
  }

  return {
    messages,
    competingView: surplusIsCompetingView(localMessages, messages, {
      optimisticMessageId: options?.excludeMessageId
    })
  }
}
