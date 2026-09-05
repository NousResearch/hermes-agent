/**
 * SESSION DOT STATE — one map from session id to the single status a surface
 * should paint, so the sidebar, the pane tabs, and the switcher can never
 * disagree about what a session is doing.
 *
 * It exists for two reasons the individual membership atoms cannot cover:
 *
 * 1. PRIORITY IN ONE PLACE. The signals overlap — a session can be working AND
 *    unread AND running a background job — and resolving that per call site is
 *    how surfaces drift apart.
 * 2. LINEAGE. Compression rotates a conversation's stored id, so a sidebar row,
 *    a persisted tile, and the route can each hold a different tip of one
 *    lineage. Every state is claimed under every alias (see `lineageAliases`),
 *    so a surface gets the right answer whichever tip it happens to hold.
 *
 * The inputs are all reference-stable across stream deltas, so this recomputes
 * on status edges rather than per token.
 *
 * Unread has TWO sources, both claiming the same state: the runtime marker
 * (a turn finished in the background while this window wasn't looking at it,
 * $unreadFinishedSessionIds — transient) and the backend's derived read-state
 * watermark (row.unread — persists across restarts and is visible to every
 * surface). The write side of the persisted flag lives in session-unread.ts.
 */

import { computed } from 'nanostores'

import { stableArray, stableRecord } from '@/lib/stable-array'

import { $backgroundRunningSessionIds } from './composer-status'
import { $messagingSessions, $sessions, $unreadFinishedSessionIds, lineageAliases } from './session'
import {
  $attentionSessionIds,
  $draftSessionIds,
  $sessionStates,
  $stalledSessionIds,
  $workingSessionIds
} from './session-states'
import { $unreadWriteGuard, UNREAD_WRITE_GUARD_MS } from './session-unread-remote'
import { $subagentsBySession, activeSubagentCount } from './subagents'

// Sessions parked in async delegation: the parent turn has ended (busy=false —
// delegate_task(background=true) returns its handle the moment the children
// are spawned) while those subagents keep working for minutes. Without this
// input the sidebar row dropped to a plain idle dot mid-delegation, reading as
// "done" while work was still running in child sessions. Same runtime→stored
// bridge and fresh-chat fallback as $backgroundRunningSessionIds:
// $subagentsBySession is keyed by runtime id, surfaces key on stored ids, and
// lineageAliases covers whichever tip of the conversation a surface holds.
let delegatingIds: readonly string[] = []
export const $delegatingSessionIds = computed(
  [$subagentsBySession, $sessionStates, $sessions],
  (bySession, states, sessions) => {
    const ids = new Set<string>()

    for (const [runtimeId, items] of Object.entries(bySession)) {
      if (activeSubagentCount(items) === 0) {
        continue
      }

      for (const alias of lineageAliases(states[runtimeId]?.storedSessionId ?? runtimeId, sessions)) {
        ids.add(alias)
      }
    }

    return (delegatingIds = stableArray(delegatingIds, [...ids]))
  }
)

export type SessionDotState =
  | 'background'
  | 'draft'
  | 'expensive'
  | 'idle'
  | 'needs-input'
  | 'stalled'
  | 'unread'
  | 'working'

/**
 * Cache-read tokens past which an otherwise-idle conversation is flagged as
 * EXPENSIVE. A long-lived chat re-reads its whole cached prefix every turn, so
 * `cache_read_tokens` climbs with conversation length regardless of the dollar
 * cost the provider quotes — which is 0 on subscription/OAuth auth, so the
 * per-session `*_cost_usd` fields cannot drive this. We flag off the token
 * count instead.
 *
 * Reference: at Anthropic Opus cache-read pricing (~$1.50 / 1M tokens),
 * 33,000,000 tokens ≈ $50 of cumulative cache re-reads. Tune here — this is the
 * one number that decides when the dot turns red.
 */
export const EXPENSIVE_CACHE_READ_TOKENS = 33_000_000

/** Whether a session row has accumulated enough cache-read tokens to be worth
 *  flagging. Degrades gracefully: an undefined count (a backend predating the
 *  field) is "no opinion", never expensive. */
export const isExpensiveByCacheReads = (cacheReadTokens?: number): boolean =>
  typeof cacheReadTokens === 'number' && cacheReadTokens >= EXPENSIVE_CACHE_READ_TOKENS

/** The sidebar row's arc. A quiet turn is still authoritatively running, so
 *  `stalled` keeps it; a blocking prompt drops it, because the amber dot is the
 *  louder cue and two treatments at once fight each other. */
export const showsRunningArc = (state: SessionDotState): boolean => state === 'stalled' || state === 'working'

/** Whether this turn is the session's own, live: brighter title, and the row's
 *  age yields to the actions menu. Wider than the arc — a turn waiting on an
 *  answer has not ended. */
export const hasLiveTurn = (state: SessionDotState): boolean => showsRunningArc(state) || state === 'needs-input'

/** The buckets the sidebar's status filter and ordering work in. `stalled` and
 *  `background` fold into the state a user would name them. */
export type SessionStatusBucket = 'draft' | 'idle' | 'needs-input' | 'unread' | 'working'

export const sessionStatusBucket = (state: SessionDotState = 'idle'): SessionStatusBucket =>
  state === 'stalled' || state === 'background'
    ? 'working'
    : // An expensive session is by construction otherwise-idle (the claim is the
      // weakest of all), so it sorts and filters as idle — the red is a hint on a
      // quiet row, not a status of its own.
      state === 'expensive'
      ? 'idle'
      : state

const STATUS_RANK: Record<SessionStatusBucket, number> = {
  'needs-input': 0,
  working: 1,
  unread: 2,
  draft: 3,
  idle: 4
}

/** Loudest first — what ordering by status sorts on. */
export const sessionStatusRank = (state?: SessionDotState): number => STATUS_RANK[sessionStatusBucket(state)]

let dotStates: Readonly<Record<string, SessionDotState>> = {}

export const $sessionDotStateById = computed(
  [
    $attentionSessionIds,
    $workingSessionIds,
    $stalledSessionIds,
    $backgroundRunningSessionIds,
    $delegatingSessionIds,
    $unreadFinishedSessionIds,
    $draftSessionIds,
    $sessions,
    $unreadWriteGuard
  ],
  (attention, working, stalled, background, delegating, unread, draft, sessions, unreadWriteGuard) => {
    const next: Record<string, SessionDotState> = {}

    const claim = (ids: readonly string[], state: SessionDotState) => {
      for (const id of ids) {
        for (const alias of lineageAliases(id, sessions)) {
          next[alias] = state
        }
      }
    }

    // Weakest claim first — each pass overwrites the one above it, so the order
    // below IS the priority order. A blocking prompt outranks everything: it is
    // the only state that needs the user.
    //
    // Expensive is weakest of ALL — even weaker than draft — because it is a
    // property of the conversation's accumulated cost, not of anything it is
    // doing right now. It surfaces ONLY on a row nothing else claims (an idle
    // session), so any real activity below speaks over it and an attention cue
    // is never masked by a cost hint. Degrades gracefully: a row whose backend
    // omits cache_read_tokens is never flagged (isExpensiveByCacheReads).
    const expensive: string[] = []

    for (const s of sessions) {
      if (isExpensiveByCacheReads(s.cache_read_tokens)) {
        expensive.push(s.id)
      }
    }

    claim(expensive, 'expensive')

    // Draft is weakest of the activity states: it says only "no turn has
    // happened here yet", so the first thing that does happen speaks over it.
    claim(draft, 'draft')
    claim(unread, 'unread')

    // Persisted read state (backend watermark): a row marked unread keeps the
    // same emerald dot a background finish would paint, and survives
    // restarts. Same tier as the runtime marker — both mean "there is
    // something here you haven't opened". A list page that predates one of
    // our own writes is fenced out by the write guard: keep OUR value until a
    // page confirms it or the guard expires.
    const persistedUnread: string[] = []

    for (const s of sessions) {
      const entry = unreadWriteGuard.get(s.id)

      if (entry && Date.now() - entry.at < UNREAD_WRITE_GUARD_MS) {
        if (entry.value) {
          persistedUnread.push(s.id)
        }

        continue
      }

      if (s.unread === true) {
        persistedUnread.push(s.id)
      }
    }

    claim(persistedUnread, 'unread')

    claim(background, 'background')
    // Async delegation: the parent turn has ended but its subagents are still
    // running, so the session's work continues in child sessions. Same visual
    // claim as background processes — and it yields to `working` below the
    // moment the parent turn itself is live (synchronous orchestrator children).
    claim(delegating, 'background')
    claim(working, 'working')

    // Stalled REFINES working rather than rivalling it — the turn is still
    // authoritatively running, it has just gone quiet — so it only downgrades a
    // session already claimed as working. The hint outlives its turn by a tick
    // on some paths; without this it could invent a running session.
    for (const id of stalled) {
      for (const alias of lineageAliases(id, sessions)) {
        if (next[alias] === 'working') {
          next[alias] = 'stalled'
        }
      }
    }

    claim(attention, 'needs-input')

    return (dotStates = stableRecord(dotStates, next))
  }
)

/** Listed, non-archived rows whose resolved status is unread. Alias keys in
 *  `$sessionDotStateById` are ignored unless they are themselves a listed row. */
export function unreadSessionCount(
  byId: Readonly<Record<string, SessionDotState>>,
  ...lists: Array<readonly { archived?: boolean; id: string }[]>
): number {
  let n = 0

  for (const rows of lists) {
    for (const row of rows) {
      if (!row.archived && byId[row.id] === 'unread') {
        n++
      }
    }
  }

  return n
}

/** The titlebar badge. Cron sessions are deliberately EXCLUDED: cron runs
 *  finish unwatched by design, so counting them turns the badge into a cron
 *  run counter that is permanently lit (#93552). Their unread state stays
 *  visible where it belongs — the sidebar's cron section rows — and
 *  "mark all as read" still acks them (ackAllSessionsRead iterates cron rows). */
export const $unreadSessionCount = computed(
  [$sessionDotStateById, $sessions, $messagingSessions],
  (byId, sessions, messaging) => unreadSessionCount(byId, sessions, messaging)
)
