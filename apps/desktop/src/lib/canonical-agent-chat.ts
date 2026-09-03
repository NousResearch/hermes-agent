/**
 * The shared canonical agent-chat identity contract.
 *
 * One agent profile has exactly ONE forever-chat: the session titled exactly
 * `CANONICAL_AGENT_CHAT_TITLE` on that profile. Identity is the NAME
 * (profile, title) pair — resolved server-side via `session.list { title,
 * include_hidden: true }` — never a stored session-id pointer. See the Bot
 * Mode section of the repo root AGENTS.md for why: five hardening waves
 * (#88690, #90732, #90751, the #91791 revert, #92042) each guarded a way a
 * stored pointer could dangle or get stolen; name-as-identity removes the
 * failure class instead of guarding it.
 *
 * This module is THE canonical identity-resolution flow — lookup,
 * lookup-failure handling, existing-session adoption, creation,
 * creation-flight/concurrency protection, title materialization/persistence,
 * title-conflict recovery and winner adoption, and fail-closed behavior
 * throughout. `plugins/hermes-bots/canonical-chat.ts` (Bot Mode's click
 * path) and `sdk/index.ts`'s `host.openCanonicalAgentChat` (the plugin-
 * facing door) both call `resolveCanonicalAgentChat` — there is exactly one
 * implementation of the flow above; callers differ only in how they perform
 * the RPCs and how they navigate once identity is resolved (injected via
 * `CanonicalAgentChatCallbacks`).
 *
 * Fail-closed is the load-bearing property throughout:
 *  - a lookup RPC failure must never be read as "no chat exists" — that
 *    reopens the exact way a forever-chat used to get forked after a lookup
 *    raced a backend restart;
 *  - a title-materialization failure that is NOT a recognized
 *    already-in-use conflict must abort the operation rather than handing
 *    back an uncanonicalized session as if it were the real thing;
 *  - a target absent from the CURRENT authorized `host.agents()` roster
 *    must never reach lookup/create/open at all — callers are required to
 *    check `isAuthorizedTarget` (or their own equivalent roster check)
 *    before calling this resolver.
 */

export const CANONICAL_AGENT_CHAT_TITLE = 'Bot Chat'

/** A `session.list` row as the registry lookup reads it. */
export interface CanonicalAgentChatRow {
  id?: string
  /** Compression-lineage tip — the live session a durable id currently maps to. */
  resolved_id?: string
  root_title?: string
  title?: string
  message_count?: number
}

/**
 * The complete source-qualified identity of an agent's canonical chat.
 * Never collapse this to a bare `{ connectionId, profile }` — `profile`
 * (the DISPLAYED/logical identity a UI selects by) and `targetProfile` (the
 * backend profile name the RPC must actually address) are two different
 * things whenever a Desktop alias or a remote connection's own profile
 * naming diverges from the local label (see routing.ts's alias-identity
 * section and `botConnectionRoute`/`backendTargetProfile`). Every RPC in
 * this flow addresses `targetProfile`; every registry/roster comparison
 * compares the full descriptor, never `profile` alone.
 */
export interface CanonicalAgentChatTarget {
  /** Null for a local/unscoped target on the active connection. */
  connectionId: string | null
  /** Displayed/logical profile — what a UI selects by, and what routing
   *  metadata is keyed on. */
  profile: string
  /** Backend profile name the RPC must actually address. Equals `profile`
   *  unless a Desktop alias or remote naming makes them diverge. */
  targetProfile: string
}

/** A `host.agents()` roster entry, reduced to what authorization needs.
 *  `targetProfile` is REQUIRED — a row that omits its own backend identity
 *  cannot vouch for any target (Architect corrective, 2026-09-02, sixth
 *  pass: an optional/defaultable `targetProfile` here let a roster row
 *  missing the field silently authorize via a substituted `profile`). */
export interface AuthorizedRosterEntry {
  connectionId?: string | null
  profile: string
  targetProfile: string
}

/** A minimal shape of the current authorized roster, as reported by
 *  `host.agents()`. Sources with a live connection but a targeted profile
 *  not present in `agents` are NOT authorized — only enumerated rows count. */
export interface AuthorizedRoster {
  agents: AuthorizedRosterEntry[]
}

function normalizeConnectionId(value: null | string | undefined): string {
  return (value ?? '').trim()
}

/** Is this EXACT source-qualified target present in the CURRENT authorized
 *  roster? An arbitrary non-blank `{connectionId, profile}` pair is never
 *  sufficient on its own — the roster is the only authority. Requires a
 *  FULL-DESCRIPTOR match against a single roster entry: `connectionId`,
 *  displayed `profile`, AND backend `targetProfile` must all agree with the
 *  same entry (Architect corrective, 2026-09-02). A mismatched
 *  `targetProfile` is rejected even when `connectionId`/`profile` otherwise
 *  look valid, and a mismatched displayed `profile` is rejected even when
 *  `connectionId`/`targetProfile` otherwise look valid — neither field may
 *  be silently collapsed into or substituted for the other. */
export function isAuthorizedCanonicalChatTarget(
  target: CanonicalAgentChatTarget,
  roster: AuthorizedRoster
): boolean {
  const connectionId = normalizeConnectionId(target.connectionId)
  const profile = (target.profile || '').trim()
  const targetProfile = (target.targetProfile || '').trim()

  if (!profile || !targetProfile) {
    return false
  }

  return (roster.agents ?? []).some(entry => {
    const entryConnectionId = normalizeConnectionId(entry.connectionId)
    const entryProfile = (entry.profile || '').trim()
    // No fallback to entry.profile: a roster row is only a valid EXACT
    // {connectionId, profile, targetProfile} admission when it explicitly
    // carries its own backend targetProfile. Substituting entry.profile
    // for a missing targetProfile let a row lacking the required third
    // field authorize a target it never actually vouched for (Architect
    // corrective, 2026-09-02, sixth pass).
    const entryTargetProfile = (entry.targetProfile || '').trim()

    if (!entryTargetProfile) {
      return false
    }

    return entryConnectionId === connectionId && entryProfile === profile && entryTargetProfile === targetProfile
  })
}

/** True when a session summary IS the canonical registry row. root_title is
 *  the durable lineage-root title reported by exact-lookup gateways; plain
 *  title covers windowed listings from older gateways. */
export function isCanonicalAgentChatRow(row: CanonicalAgentChatRow): boolean {
  const rootTitle = String(row?.root_title || '').trim()
  const title = String(row?.title || '').trim()

  return rootTitle === CANONICAL_AGENT_CHAT_TITLE || (!rootTitle && title === CANONICAL_AGENT_CHAT_TITLE)
}

/** Default classifier for a title-write rejection meaning "another writer
 *  already holds the canonical title" (adopt-before-mint territory) vs a
 *  genuine failure. Matches the established Bot Mode convention. */
export function isTitleConflictError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String((error as { message?: unknown })?.message ?? error ?? '')

  return /already in use/i.test(message)
}

export interface CanonicalAgentChatCallbacks {
  /** THE identity lookup, scoped to `target`'s own source. MUST throw (not
   *  resolve to a sentinel) on any RPC failure — the resolver's fail-closed
   *  guarantee depends on a thrown lookup propagating rather than reading
   *  as "no chat exists yet". */
  lookup(target: CanonicalAgentChatTarget): Promise<CanonicalAgentChatRow | null>
  /** Create the lazy canonical session. `runtimeId` is used to issue the
   *  title write; `storedId` is what a caller ultimately opens/returns as
   *  identity when titling cannot run (older-gateway compat only, see
   *  `onTitleFailure`). Either may be absent depending on gateway version. */
  create(target: CanonicalAgentChatTarget): Promise<{ runtimeId: null | string; storedId: null | string }>
  /** Persist the canonical title on `runtimeId`. MUST throw on any
   *  failure — implementations should throw an error whose message the
   *  configured `isTitleConflict` recognizes when another writer has
   *  already taken the title. */
  titleSession(target: CanonicalAgentChatTarget, runtimeId: string): Promise<void>
  /** Open an EXISTING canonical row — the adoption path and the
   *  winner-adoption path after a title conflict both go through this.
   *  Receives the resolved lineage-tip id to actually focus and the full
   *  row for context (message_count, etc.); implementations own
   *  compression-lineage-aware navigation and any other established
   *  open/workspace semantics. `canNavigate` reflects the caller's own
   *  staleness probe (`options.openingStillCurrent`) — implementations
   *  decide for themselves whether/how to act on it; the resolver always
   *  invokes this callback so side effects that must happen regardless of
   *  staleness (e.g. Bot Mode's compat-path persistence write) are not
   *  silently skipped. */
  openExisting(target: CanonicalAgentChatTarget, openedId: string, row: CanonicalAgentChatRow, canNavigate: boolean): Promise<void>
  /** Open a freshly created (or compat-path) session by its stored id.
   *  Same `canNavigate` contract as `openExisting`. */
  openFresh(target: CanonicalAgentChatTarget, storedId: string, canNavigate: boolean): Promise<void>
}

export interface CanonicalAgentChatOptions {
  /** Classifies a titleSession rejection as a title conflict (re-lookup +
   *  adopt the winner) vs a genuine failure. Defaults to
   *  `isTitleConflictError`. */
  isTitleConflict?: (error: unknown) => boolean
  /** Legacy-gateway compatibility escape hatch — Bot Mode's documented
   *  fallback for gateways that reject `session.title` for reasons OTHER
   *  than a title conflict (the write is simply unsupported): the lazy
   *  session persists once its first message lands, so identity is backed
   *  by `storedId` rather than a materialized title. Called ONLY when
   *  `titleSession` fails with a non-conflict error. Returning `true`
   *  tells the resolver to proceed with `storedId` as identity; returning
   *  `false` (or omitting this callback entirely) means the resolver
   *  FAILS CLOSED and rejects rather than handing back an uncanonicalized
   *  session. */
  allowUntitledCompat?: (target: CanonicalAgentChatTarget, error: unknown) => Promise<boolean> | boolean
  /** Staleness probe consulted before every navigation — when the caller
   *  has already moved on (a click-path UI concern), identity resolution
   *  still completes registry-side but the resolver must not navigate. */
  openingStillCurrent?: (() => boolean) | null
}

export interface CanonicalAgentChatResult {
  /** Durable registry id — names the chat. */
  registryId: string
  /** The id that actually took focus (the lineage tip when compressed). */
  openedId: string
}

interface CanonicalChatFlight {
  run: Promise<{ result: CanonicalAgentChatResult | null; row: CanonicalAgentChatRow | null }>
}

/** In-flight creations, keyed by caller-supplied identity key (typically
 *  `${connectionId}::${targetProfile}`). Owned HERE — the one shared
 *  concurrency guard for every caller, so two callers racing the same
 *  target's first open can never mint two canonical chats between them. */
const inflightCreations = new Map<string, CanonicalChatFlight>()

function targetKey(target: CanonicalAgentChatTarget): string {
  return `${normalizeConnectionId(target.connectionId)}::${target.targetProfile}`
}

function canNavigate(options: CanonicalAgentChatOptions): boolean {
  return !options.openingStillCurrent || options.openingStillCurrent()
}

/** THE canonical identity-resolution flow, shared by every caller. Resolves
 *  `target`'s one forever-chat: adopts it if it exists, creates it
 *  (fail-closed, conflict-aware, concurrency-protected) if it doesn't.
 *
 *  Callers MUST validate `target` against the current authorized
 *  `host.agents()` roster (`isAuthorizedCanonicalChatTarget`) before calling
 *  this — an arbitrary non-blank target is not itself authorization, and
 *  this function performs no roster check of its own since only the caller
 *  has a live `host.agents()` result to check against. */
export async function resolveCanonicalAgentChat(
  target: CanonicalAgentChatTarget,
  callbacks: CanonicalAgentChatCallbacks,
  options: CanonicalAgentChatOptions = {}
): Promise<CanonicalAgentChatResult | null> {
  const isConflict = options.isTitleConflict ?? isTitleConflictError
  const key = targetKey(target)

  // Adoption is always attempted first, outside any flight — the common
  // case (chat already exists) never touches the concurrency map at all.
  const existing = await callbacks.lookup(target)

  if (existing?.id) {
    const openedId = String(existing.resolved_id || existing.id)

    await callbacks.openExisting(target, openedId, existing, canNavigate(options))

    return { registryId: String(existing.id), openedId }
  }

  // No existing row: dedupe concurrent creation by target key. A second
  // caller arriving while the first is still creating adopts the first's
  // result rather than racing session.create — that race is exactly how a
  // double-click used to mint two canonical chats. The joiner still opens
  // explicitly for itself (never assumes the first caller's navigation
  // satisfies it), using openFresh when the flight created and openExisting
  // when the flight found/adopted a row (including the winner of a title
  // conflict) — matching each callback's own documented semantics.
  //
  // `canNavigate(options)` is evaluated FRESH at each point of use, never
  // cached — a caller's own staleness probe can flip mid-flight (the click
  // moved on while creation was still in progress), and Bot Mode's
  // established navigation semantics depend on observing that live value at
  // the moment of the actual open, not at call start.
  const inflight = inflightCreations.get(key)

  if (inflight) {
    return inflight.run.then(async ({ result, row }) => {
      if (!result) {
        return null
      }

      if (row) {
        await callbacks.openExisting(target, result.openedId, row, canNavigate(options))
      } else {
        await callbacks.openFresh(target, result.openedId, canNavigate(options))
      }

      return result
    })
  }

  const flight: CanonicalChatFlight = {
    run: (async (): Promise<{ result: CanonicalAgentChatResult | null; row: CanonicalAgentChatRow | null }> => {
      const created = await callbacks.create(target)
      const { runtimeId, storedId } = created

      if (!runtimeId && !storedId) {
        // Nothing was actually created — never claim identity for a no-op.
        return { result: null, row: null }
      }

      if (runtimeId) {
        try {
          await callbacks.titleSession(target, runtimeId)
        } catch (error) {
          if (isConflict(error)) {
            // ADOPT-BEFORE-MINT: another writer took the canonical title
            // between our registry miss and this write. Re-consult and
            // adopt the winner rather than proceeding into our own stray
            // (zero-message) session, which would fork the forever-chat.
            const winner = await callbacks.lookup(target)

            if (winner?.id) {
              const openedId = String(winner.resolved_id || winner.id)

              await callbacks.openExisting(target, openedId, winner, canNavigate(options))

              return { result: { registryId: String(winner.id), openedId }, row: winner }
            }

            // Conflict reported but re-lookup found nothing canonical —
            // fail closed rather than guessing.
            throw new Error('Canonical Bot Chat title conflict could not be resolved by re-lookup')
          }

          // Not a recognized conflict: fail closed UNLESS the caller
          // explicitly opted into the legacy-gateway compat path.
          const compat = (await options.allowUntitledCompat?.(target, error)) ?? false

          if (!compat) {
            throw error instanceof Error
              ? error
              : new Error(`Could not establish the canonical Bot Chat title: ${String(error)}`)
          }
          /* compat: the lazy row persists once its first message lands;
             fall through to open storedId as identity below. */
        }
      }

      if (!storedId) {
        // Titled successfully but no stored id was returned — the RPC
        // contract was not honored; fail closed rather than returning an
        // unresolvable identity.
        throw new Error('Canonical Bot Chat creation did not return a stored session id')
      }

      await callbacks.openFresh(target, storedId, canNavigate(options))

      return { result: { registryId: storedId, openedId: storedId }, row: null }
    })().finally(() => {
      if (inflightCreations.get(key) === flight) {
        inflightCreations.delete(key)
      }
    }),
  }

  inflightCreations.set(key, flight)

  return flight.run.then(({ result }) => result)
}
