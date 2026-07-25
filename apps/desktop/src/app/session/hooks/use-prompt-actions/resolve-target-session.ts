import { resolveSessionProfile } from '../use-session-actions/utils'

import type { GatewayRequest } from './utils'

/**
 * Resolve the runtime session a submit or slash command must target.
 *
 * Both prompt pipelines need this answer and they must agree: `submit.ts`
 * sends the user's text into it, and `slash.ts` runs backend commands against
 * it. Per-session backend state — `/goal` (persisted in SessionDB `state_meta`
 * under `goal:<session_id>`), usage, status, yolo — is keyed by that id, so a
 * pipeline that resolves it differently reads and writes a DIFFERENT session's
 * state than the one the user is looking at.
 *
 * That is the bug this exists to prevent. `slash.ts` used to resolve with a
 * bare `hint || activeRef || createSession()`, so whenever the runtime binding
 * was momentarily absent — profile swap, reconnect, orphan-reap, request
 * timeout — a slash command silently MINTED A NEW SESSION and ran against it.
 * `/goal <text>` then set a goal on the chat, and a following `/goal status`
 * reported "No active goal" because it was asking a session that had just been
 * created. Same hole for every exec/rpc command (`/usage`, `/status`,
 * `/tools`, …), not just `/goal`.
 *
 * The ladder (highest-trust rung first), mirroring `submit.ts`:
 *
 *   1. An explicit runtime id from the caller (queue drain / tile) — always
 *      authoritative.
 *   2. The live runtime ref. A durable ROUTE outranks it: when the URL names a
 *      conversation whose runtime binding the cache does not confirm, the ref
 *      is stale or cross-wired (often from the previous profile) and must not
 *      capture the command. With no route in play the ref is authoritative.
 *   3. `session.resume` on the routed (else selected) stored session,
 *      re-registered on the profile that OWNS it — never whichever profile
 *      happens to be live, which forks the conversation into the wrong
 *      state.db (#67603).
 *   4. Only a genuine new-chat draft — no durable session in play at all —
 *      creates a session.
 *
 * Returns null when a durable conversation was targeted but its runtime could
 * not be rebound. Callers surface that instead of silently retargeting: a
 * command that runs against the wrong session is worse than one that reports
 * it could not run.
 */
export interface ResolveTargetSessionDeps {
  activeRuntimeId: null | string
  createSession: () => Promise<null | string>
  explicitRuntimeId?: null | string
  getRuntimeIdForStoredSession: (storedSessionId: string) => null | string
  requestGateway: GatewayRequest
  routedStoredSessionId: null | string
  selectedStoredSessionId: null | string
}

export async function resolveTargetSessionId(deps: ResolveTargetSessionDeps): Promise<null | string> {
  const {
    activeRuntimeId,
    createSession,
    explicitRuntimeId,
    getRuntimeIdForStoredSession,
    requestGateway,
    routedStoredSessionId,
    selectedStoredSessionId
  } = deps

  // 1. An explicit target always wins — the caller knows which session it means.
  if (explicitRuntimeId) {
    return explicitRuntimeId
  }

  // A route whose runtime binding is incomplete or cross-wired outranks the
  // live ref: a profile swap / reconnect can leave the previous profile's
  // runtime active while the URL still names the conversation on screen.
  // Matches submit.ts's `routedSessionNeedsResume`.
  const routedNeedsResume = Boolean(
    routedStoredSessionId &&
    (selectedStoredSessionId !== routedStoredSessionId ||
      !activeRuntimeId ||
      activeRuntimeId !== getRuntimeIdForStoredSession(routedStoredSessionId))
  )

  // 2. Trust the live runtime unless the durable route disagrees with it.
  if (activeRuntimeId && !routedNeedsResume) {
    return activeRuntimeId
  }

  // 3. Rebind the durable conversation on its owning profile. The route wins
  //    over a stale selection; otherwise continue whatever is selected.
  const storedTarget = routedNeedsResume ? routedStoredSessionId : (selectedStoredSessionId ?? routedStoredSessionId)

  if (storedTarget) {
    try {
      const profile = await resolveSessionProfile(storedTarget)

      const resumed = await requestGateway<{ session_id?: string }>('session.resume', {
        session_id: storedTarget,
        source: 'desktop',
        ...(profile ? { profile } : {})
      })

      return resumed?.session_id || null
    } catch {
      // A targeted durable conversation whose runtime cannot be rebound must
      // NOT fall through to createSession() — that is precisely the fork this
      // resolver exists to prevent (#55578 class).
      return null
    }
  }

  // 4. A genuine new-chat draft: nothing durable is in play.
  return activeRuntimeId || (await createSession())
}
