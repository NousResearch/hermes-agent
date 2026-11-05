import {
  extendRefreshPageToOverlap,
  graftRefreshedTailOntoBackfill,
  olderPageReader
} from '@/app/chat/transcript-backfill'
import { getLatestSessionMessages, type ProfileScope } from '@/hermes'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import {
  type InFlightRecoveryResult,
  journaledTurnFinished,
  type OwnerPendingTurns,
  ownerPendingTurns,
  runningAdmission
} from '@/lib/inflight-turn-journal'
import type { SessionMessage, SessionMessagesResponse, SessionResumeResult } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

import { reconcileDurableHistory } from './utils'

export interface GraftedTranscriptPage {
  page: SessionMessagesResponse
  /** The REST page grafted onto the view's older backfilled pages. */
  transcript: ChatMessage[]
  /** That transcript reconciled with the view's local rows. */
  snapshot: ChatMessage[]
}

export interface ResumeBarrierReread {
  /** The page read again after the barrier, when one was needed and read. */
  page: GraftedTranscriptPage | null
  /** The owner's pending list for journal recovery, marked `replyUnread` when a finished turn's page could not be read
   *  again, so its journal folds as settled output instead of being retired beside a page that may lack the reply. */
  owner: OwnerPendingTurns | undefined
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
 * more. So is a page beside a journaled turn the owner no longer lists, idle or
 * not (a queued successor may already run): that turn settled before this
 * snapshot, after its reply rows were written, but possibly after the page was
 * read, and resume retires its journal, so nothing local would stand in for a
 * missing reply.
 */
export async function rereadPageAfterResumeBarrier(
  resumed: Pick<SessionResumeResult, 'inflight' | 'messages' | 'pending' | 'queued'> & { running?: boolean },
  early: GraftedTranscriptPage | null,
  storedSessionId: string,
  scope: ProfileScope
): Promise<ResumeBarrierReread> {
  const owner = ownerPendingTurns(resumed.pending)
  const tail = early?.snapshot.at(-1)

  if (!early || !tail || resumed.messages?.length) {
    return { page: null, owner }
  }

  const idleWithLocalTail =
    resumed.running === false &&
    !resumed.inflight &&
    !resumed.queued &&
    !early.transcript.some(message => message.id === tail.id)

  if (!idleWithLocalTail && !journaledTurnFinished(storedSessionId, owner)) {
    return { page: null, owner }
  }

  const page = await getLatestSessionMessages(storedSessionId, scope).catch(() => null)

  if (!page || (early.page.session_id && page.session_id && page.session_id !== early.page.session_id)) {
    return { page: null, owner: owner && { ...owner, replyUnread: true } }
  }

  return { page: await graftTranscriptPage(page, early.snapshot, storedSessionId, scope), owner }
}

/** The session fields a resume takes from its in-flight journal recovery. `latest` is the session as the resume
 *  commits; `atStart` holds every session's state from before the resume was requested. */
export function recoveredTurnFields(
  recovery: InFlightRecoveryResult,
  resumed: Pick<SessionResumeResult, 'pending' | 'session_id' | 'turn_started_at'>,
  running: boolean,
  latest: Pick<ClientSessionState, 'turnAdmissionId'>,
  atStart: ReadonlyMap<string, Pick<ClientSessionState, 'turnAdmissionId'>>
) {
  // The owner reports when the in-flight turn started, so the clock resumes instead of resetting to 0:00.
  const resumedTurnStartedAt =
    typeof resumed.turn_started_at === 'number' && resumed.turn_started_at > 0 ? resumed.turn_started_at * 1000 : null

  const answered = recovery.admissionId ?? null

  return {
    // The execution a recovered journal still names, or none: the next stamped frame of a live turn names it.
    turnAdmissionId: admissionAfterResume(latest, atStart.get(resumed.session_id) ?? {}, answered, resumed.pending),
    ...(recovery.applied
      ? {
          sawAssistantPayload: true,
          // Point live deltas at the recovered row when the backend is
          // still mid-turn; a settled recovery keeps the stream idle.
          streamId: running ? recovery.streamId : null,
          turnStartedAt: running ? (recovery.turnStartedAt ?? resumedTurnStartedAt) : null
        }
      : { turnStartedAt: running && resumedTurnStartedAt !== null ? resumedTurnStartedAt : null })
  }
}

/** The execution a warm activation leaves the session naming. A frame that named one while `session.activate` was
 *  in flight is newer than its snapshot; otherwise the snapshot's running execution replaces the one this tab last
 *  streamed, which may have finished before the session moved on elsewhere. */
export function activatedTurnAdmission(
  latest: Pick<ClientSessionState, 'turnAdmissionId'>,
  atActivateStart: Pick<ClientSessionState, 'turnAdmissionId'>,
  pending: unknown
): null | string {
  return admissionAfterResume(latest, atActivateStart, runningAdmission(pending), pending)
}

/** A frame that named an execution while a resume was in flight is newer than `answered`, what the resume names.
 *  Without one, the session keeps the admission it already names while the owner still lists it: a turn lost in a
 *  crash (`unknown`) runs no execution, but its journal still needs it. */
function admissionAfterResume(
  latest: Pick<ClientSessionState, 'turnAdmissionId'>,
  atStart: Pick<ClientSessionState, 'turnAdmissionId'>,
  answered: null | string,
  pending: unknown
): null | string {
  const named = latest.turnAdmissionId ?? null

  if (named !== (atStart.turnAdmissionId ?? null)) {
    return named
  }

  return answered ?? (named && ownerPendingTurns(pending)?.admissionIds.has(named) ? named : null)
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
