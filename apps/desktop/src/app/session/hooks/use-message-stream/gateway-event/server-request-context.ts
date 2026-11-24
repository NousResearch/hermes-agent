import { JSON_RPC_INTERNAL_ERROR } from '@hermes/shared'

import type { ScopedServerRequest } from '@/store/gateway'

import type { GatewayEventDeps } from './types'

// What the server-request handler modules share: a handler's context and the guards of the
// blocking-input family. `server-requests.ts` owns the method registry and the dispatch.

export const str = (v: unknown): string => (typeof v === 'string' ? v : '')

export interface ServerRequestContext {
  deps: Pick<
    GatewayEventDeps,
    'activeSessionIdRef' | 'sessionInterrupted' | 'sessionStateByRuntimeIdRef' | 'updateSessionState' | 'upsertToolCall'
  >
  request: ScopedServerRequest
  /** The session the request names ('' when unscoped). */
  sessionId: string
  /** The named session is the one on screen. */
  isActiveSession: boolean
}

export type Handler = (ctx: ServerRequestContext) => void

export const markNeedsInput = (ctx: ServerRequestContext) => {
  if (ctx.sessionId) {
    ctx.deps.updateSessionState(ctx.sessionId, state => ({ ...state, needsInput: true }))
  }
}

/**
 * A blocking-input card must not park for a session whose runtime is already
 * interrupted — the user hit Stop, or `removeSession` marked the doomed runtime
 * interrupted before it deletes the row. A frame still in flight would otherwise
 * re-create an overlay (and native notification) for a turn that is gone
 * (#75587). Answer it rather than drop it: the backend blocks on this frame, and
 * an error reply is the same "unanswered" its own `request.cancel` produces, so
 * the tool returns now instead of waiting out its deadline. Sessionless requests
 * (app-level Bot Screen install) are never gated.
 */
export const declineIfSessionStopped = (ctx: ServerRequestContext): boolean => {
  if (!ctx.sessionId || !ctx.deps.sessionInterrupted(ctx.sessionId)) {
    return false
  }

  ctx.request.fail(JSON_RPC_INTERNAL_ERROR, 'session interrupted')

  return true
}
