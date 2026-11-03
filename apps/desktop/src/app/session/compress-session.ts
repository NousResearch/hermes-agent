import { mergeUsage } from '@hermes/shared'

import { mergeOlderTranscriptPage } from '@/app/chat/transcript-backfill'
import { invalidateContextBreakdown } from '@/app/shell/hooks/use-context-breakdown'
import { toChatMessages } from '@/lib/chat-messages'
import { setCurrentUsage, setSessions } from '@/store/session'
import { requestForOwnedSession } from '@/store/session-states'

import type { ClientSessionState, SessionCompressResponse } from '../types'

import { type GatewayRequest, withSessionNotFoundResume } from './hooks/use-prompt-actions/utils'

// Manual compression is LLM-bound and routinely outlives the desktop's 30s
// default WS request timeout on large sessions. The gateway blocks its own
// compute-host wait for up to compression.context_total_ceiling_seconds + 30s
// (capped at 630s, tui_gateway/server.py _COMPUTE_HOST_COMPRESS_WAIT_CAP_SECS)
// and then answers `status: 'pending'` rather than an error, so this budget
// must sit above that cap or the desktop reports a false timeout while the
// host is still compressing (#97948).
export const SESSION_COMPRESS_TIMEOUT_MS = 660_000

export interface CompressSessionDeps {
  /** True when `sessionId` is the chat on screen. Only then may the shared
   *  usage readouts be repainted — a compress that lands after the user moved
   *  to another chat must not paint its numbers under that chat's name. */
  isActive?: (sessionId: string) => boolean
  /** Called with the re-minted runtime id after a stale-runtime recovery. */
  onRecovered?: (liveSessionId: string) => void
  /** Dispatch for the session's own backend. Callers route by owner
   *  (`requestForOwnedSession`) or by the invocation's destination; this
   *  module never picks a socket itself. */
  request: GatewayRequest
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: null | string
  ) => unknown
}

export interface CompressSessionTarget {
  focusTopic?: string
  /** Runtime id of the conversation to compress. Captured by the caller, so a
   *  compress started on this chat finishes on it even if the user navigates. */
  sessionId: string
  storedSessionId: null | string
}

export type CompressSessionOutcome =
  /** The gateway's compute-host wait expired while the host still compresses;
   *  it pushes the completion edge itself, so nothing below applies. */
  | { kind: 'deferred'; message: string; sessionId: string }
  /** The backend answered but compressed NOTHING: the in-process route's
   *  `compressed: false` + `lock_held` skip, a canonical `noop` report, or a
   *  read-only `preview`. Reporting one of these as a success told the user
   *  their conversation had been summarized when its transcript never moved. */
  | { kind: 'skipped'; message: string; result: SessionCompressResponse; sessionId: string }
  | {
      aborted: boolean
      kind: 'done'
      /** Human line(s): the backend's summary, its host ack, or the message
       *  count. Empty only if the backend sent nothing at all. */
      message: string
      result: SessionCompressResponse
      sessionId: string
    }

/** Bind a request to the gateway that OWNS `sessionId` (tile route → owner
 *  hint → connection-tagged row). The ambient dispatcher re-scopes `profile`
 *  to the FOCUSED chat, which is the wrong backend as soon as the user
 *  switches away from the conversation they clicked on. */
export function ownedSessionRequest(sessionId: string, ambientRequest: GatewayRequest): GatewayRequest {
  return (method, params, timeoutMs) => requestForOwnedSession(sessionId, ambientRequest, method, params, timeoutMs)
}

/** Compress one conversation. The single implementation behind both the typed
 *  `/compress` command and the context-usage popover's action: one RPC (with
 *  the stale-runtime recovery every session-scoped call needs), one transcript
 *  rewrite, one usage/title refresh, one breakdown invalidation. Callers own
 *  their own notices and any chat-visible record of the result. */
export async function compressSession(
  deps: CompressSessionDeps,
  target: CompressSessionTarget
): Promise<CompressSessionOutcome> {
  const focusTopic = target.focusTopic?.trim() ?? ''

  // NOT alsoTimeout: compress is legitimately LLM-slow, so a timeout here must
  // not be retried as a dead session.
  const { result, sessionId } = await withSessionNotFoundResume(
    target.sessionId,
    target.storedSessionId,
    liveSessionId =>
      deps.request<SessionCompressResponse>(
        'session.compress',
        { session_id: liveSessionId, ...(focusTopic ? { focus_topic: focusTopic } : {}) },
        SESSION_COMPRESS_TIMEOUT_MS
      ),
    { onRecovered: deps.onRecovered, requestGateway: deps.request }
  )

  if (result?.status === 'pending') {
    return { kind: 'deferred', message: result.message?.trim() ?? '', sessionId }
  }

  const skipped = skippedCompressionMessage(result)

  if (skipped !== null) {
    return { kind: 'skipped', message: skipped, result: result ?? {}, sessionId }
  }

  applyCompressedTranscript(result, target.storedSessionId, { ...deps, sessionId })

  const { aborted, message } = compressResultMessage(result)

  return { aborted, kind: 'done', message, result, sessionId }
}

/** The backend's own line for a reply that compressed nothing, or null when the
 *  reply really did commit a compression.
 *
 *  Keyed on the answers the two live routes actually send: the in-process route
 *  (`tui_gateway/methods_session.py`) answers `{compressed: false, lock_held:
 *  true, message}` when another process holds the compression lock, and
 *  `status: 'noop'`/`summary.noop` when nothing was removable; the canonical
 *  owner (`session.mutate compress`) answers `status: 'preview'` for the
 *  read-only report. A committed canonical receipt carries counts and a summary
 *  and no `status` at all, so it must keep falling through to `done`. */
function skippedCompressionMessage(result: SessionCompressResponse | undefined): null | string {
  if (result?.compressed === false) {
    return result.message?.trim() || 'compression skipped'
  }

  if (result?.status === 'noop' || result?.summary?.noop === true) {
    return summaryLines(result) || result?.message?.trim() || 'nothing to compress'
  }

  if (result?.status === 'preview') {
    return result.host_ack?.output?.trim() || summaryLines(result) || 'compression preview'
  }

  return null
}

/** Replace the transcript with the post-compress history so the summarized
 *  bubbles actually disappear, then refresh the figures the rewrite changed.
 *  `messages` is the same shape session.resume returns (_history_to_messages),
 *  so toChatMessages handles it directly. */
function applyCompressedTranscript(
  result: SessionCompressResponse | undefined,
  storedSessionId: null | string,
  deps: CompressSessionDeps & { sessionId: string }
) {
  const { isActive, sessionId, updateSessionState } = deps

  // The pre-compression segment stays reachable in the active session
  // (#105256): the rewrite shares no anchor with the live transcript, so
  // grafting would drop everything before the summary. Prepend it instead
  // (deduped by row/message id), keeping the post-compress history
  // authoritative for overlapping rows.
  if (Array.isArray(result?.messages)) {
    const messages = result.messages

    updateSessionState(
      sessionId,
      state => ({ ...state, messages: mergeOlderTranscriptPage(toChatMessages(messages), state.messages) }),
      storedSessionId
    )
  }

  // The transcript just shrank by 5-10x outside any turn (busy never flipped),
  // so the keyed context breakdown is now wrong by that factor. Bump the
  // invalidation generation: the statusbar gauge refetches immediately instead
  // of serving the pre-compression figure until the session is switched
  // (#94001).
  invalidateContextBreakdown(sessionId)

  const usage = { ...result?.usage, ...result?.info?.usage }

  if (Object.keys(usage).length && (isActive?.(sessionId) ?? true)) {
    setCurrentUsage(current => mergeUsage(current, usage))
  }

  if (result?.info?.title !== undefined) {
    const title = result.info.title || null

    setSessions(prev => prev.map(session => (session.id === sessionId ? { ...session, title } : session)))
  }
}

/** The line the user reads and the chat records. Aborted compressions keep
 *  their summary text but are reported as a failure. */
function compressResultMessage(result: SessionCompressResponse | undefined): { aborted: boolean; message: string } {
  const aborted = result?.status === 'aborted' || result?.summary?.aborted === true

  const lines = summaryLines(result)

  if (lines) {
    return { aborted, message: lines }
  }

  const hostOutput = result?.host_ack?.output?.trim()

  if (hostOutput) {
    return { aborted, message: hostOutput }
  }

  const removed = result?.removed ?? 0

  return { aborted, message: removed > 0 ? `compressed ${removed} messages` : 'nothing to compress' }
}

/** The summary's own lines, newline-joined; empty when the backend sent none. */
function summaryLines(result: SessionCompressResponse | undefined): string {
  return [result?.summary?.headline, result?.summary?.token_line, result?.summary?.note]
    .filter((line): line is string => Boolean(line))
    .join('\n')
}
