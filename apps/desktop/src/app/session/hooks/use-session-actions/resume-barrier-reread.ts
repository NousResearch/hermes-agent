import {
  extendRefreshPageToOverlap,
  graftRefreshedTailOntoBackfill,
  olderPageReader
} from '@/app/chat/transcript-backfill'
import { getLatestSessionMessages, type ProfileScope } from '@/hermes'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage, SessionMessagesResponse, SessionResumeResult } from '@/types/hermes'

import { reconcileDurableHistory } from './utils'

export interface GraftedTranscriptPage {
  page: SessionMessagesResponse
  /** The REST page grafted onto the view's older backfilled pages. */
  transcript: ChatMessage[]
  /** That transcript reconciled with the view's local rows. */
  snapshot: ChatMessage[]
}

/** Graft a newest-tail REST page onto the view: older pages kept, local tail reconciled. */
export async function graftTranscriptPage(
  page: SessionMessagesResponse,
  previousMessages: ChatMessage[],
  storedSessionId: string,
  scope: ProfileScope
): Promise<GraftedTranscriptPage> {
  // A long turn can push every rendered row off the newest page; read older
  // pages until they overlap so the graft keeps earlier history.
  const tail = await extendRefreshPageToOverlap(
    toChatMessages(page.messages),
    previousMessages,
    olderPageReader(storedSessionId, scope, page)
  )

  const transcript = graftRefreshedTailOntoBackfill(tail, previousMessages)

  return { page, transcript, snapshot: reconcileDurableHistory(transcript, previousMessages) }
}

/**
 * The cold resume reads REST concurrently with `session.resume`, so that read
 * can land before the turn commits. The early page then lacks the committed
 * reply, and the view keeps its local stream row, stuck at the last streamed
 * chunk. A resume carrying its snapshot settles that by row recency
 * (`prefetchServesResume`); one that omits it (`omit_messages` honoured) only
 * says the turn is idle. That idle answer is the barrier (the reply is durable
 * once it returns), so a snapshot whose tail is still a local row is read once
 * more.
 */
export async function rereadPageAfterResumeBarrier(
  resumed: Pick<SessionResumeResult, 'inflight' | 'messages' | 'queued'> & { running?: boolean },
  early: GraftedTranscriptPage | null,
  storedSessionId: string,
  scope: ProfileScope
): Promise<GraftedTranscriptPage | null> {
  const tail = early?.snapshot.at(-1)

  if (!early || !tail || resumed.running !== false || resumed.inflight || resumed.queued || resumed.messages?.length) {
    return null
  }

  if (early.transcript.some(message => message.id === tail.id)) {
    return null
  }

  const page = await getLatestSessionMessages(storedSessionId, scope).catch(() => null)

  if (!page || (early.page.session_id && page.session_id && page.session_id !== early.page.session_id)) {
    return null
  }

  return graftTranscriptPage(page, early.snapshot, storedSessionId, scope)
}

const durableRowId = ({ id, row_id }: SessionMessage): number | undefined => {
  const value = row_id ?? id

  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

const newestDurableRowId = (messages: readonly SessionMessage[] | null | undefined): number =>
  (messages ?? []).reduce((newest, message) => Math.max(newest, durableRowId(message) ?? -Infinity), -Infinity)

/**
 * The resume snapshot is newer than the cold prefetch page when it carries a
 * durable row the page had not stored yet. Frames that landed while the resume
 * was in flight (the authority pushes a reconnected viewer's missed tail before
 * it answers) are already in that snapshot, while the page was read before the
 * turn committed: painting the page would put the stream row back at its first
 * chunk. Rows are compared by durable id, never by which request returned first.
 */
function resumeSnapshotSupersedesPage(
  resumedMessages: readonly SessionMessage[] | undefined,
  pageMessages: readonly SessionMessage[] | null | undefined
): boolean {
  const pageNewest = newestDurableRowId(pageMessages)

  return Number.isFinite(pageNewest) && newestDurableRowId(resumedMessages) > pageNewest
}

/** The cold prefetch page still speaks for the resumed session: same session, and no newer than its snapshot. */
export function prefetchServesResume(
  pageSessionId: string | null,
  resumed: Pick<SessionResumeResult, 'messages' | 'resumed' | 'session_key'>,
  page: SessionMessagesResponse | null
): boolean {
  const resumedSessionId = resumed.session_key || resumed.resumed
  const sameSession = !pageSessionId || !resumedSessionId || pageSessionId === resumedSessionId

  return sameSession && !resumeSnapshotSupersedesPage(resumed.messages, page?.messages)
}
