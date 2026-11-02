import { getSessionMessages } from '@/api/sessions'
import type { SessionMessage } from '@/types/hermes'

import { listedSubmissions } from './admitted-submissions'
import { isFreshDraftScope } from './composer'

// A window can close after the backend admitted a send but before the submit ack came back. Its
// outbox entry then outlives a send that DID land: re-offering it as the draft would show the words
// twice (in the chat and in the composer). The owner stamps the admission's `submission_id` on the
// user row the turn persists (gateway/session_display.py), so the stored transcript answers
// "did it land" for that exact send, never by matching text. A send admitted but not run yet has no
// row: the owner's pending snapshot lists it by the same id instead.

const PAGE = 200
// Pages are read newest first until the send is found or the rows predate it. The margin absorbs
// clock skew between this machine and a remote backend; the page cap bounds a pathological transcript.
const SKEW_MS = 10 * 60_000
const MAX_PAGES = 20
// Opening the session resumes it, and the resume snapshot lists its pending admissions.
const SNAPSHOT_WAIT_MS = 5000

function submissionOf(message: SessionMessage): unknown {
  let metadata: unknown = message.display_metadata

  if (typeof metadata === 'string') {
    try {
      metadata = JSON.parse(metadata)
    } catch {
      return undefined
    }
  }

  return metadata && typeof metadata === 'object' ? (metadata as Record<string, unknown>).submission_id : undefined
}

const predates = (message: SessionMessage, sentAt: number) =>
  typeof message.timestamp === 'number' && message.timestamp * 1000 < sentAt - SKEW_MS

/** True only when the owner holds `submissionId` for `scope`: listed pending (read first: an admission
 *  that leaves the pending set has persisted its row by then), or its user row stored. No snapshot, a
 *  failed read, a scope that is not a stored session yet, or an exhausted search answers false: the
 *  entry is then re-offered as the draft, so an unknown outcome can show the words twice but never
 *  lose them. */
export async function sendLanded(scope: string, submissionId: string, sentAt?: number): Promise<boolean> {
  if (!scope || isFreshDraftScope(scope)) {
    return false
  }

  if ((await listedSubmissions(scope, SNAPSHOT_WAIT_MS))?.has(submissionId)) {
    return true
  }

  try {
    for (let page = 0; page < MAX_PAGES; page += 1) {
      const { messages, pagination } = await getSessionMessages(
        scope,
        undefined,
        { includeCompacted: true, limit: PAGE, offset: page * PAGE, order: 'latest' },
        { passive: true }
      )

      if (messages.some(message => message.role === 'user' && submissionOf(message) === submissionId)) {
        return true
      }

      const exhausted = !pagination || messages.length < PAGE

      if (exhausted || (sentAt !== undefined && messages.some(message => predates(message, sentAt)))) {
        return false
      }
    }
  } catch {
    return false
  }

  return false
}
