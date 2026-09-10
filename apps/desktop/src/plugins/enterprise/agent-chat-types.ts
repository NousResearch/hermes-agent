/**
 * Command Center — direct agent chat (Mode 2) types.
 *
 * Bounded release: "Command Center — Two Owner Communication Modes"
 * (2026-09-01) + its narrow SDK-seam amendment (same date). Direct agent
 * eligibility comes ONLY from the current authorized Desktop Bot Mode
 * roster (`host.agents()`) — never a hardcoded list, guessed profile name,
 * or filesystem scan. A roster entry that cannot be verified renders as
 * unavailable/no-access/unknown, never as a working card.
 */

export type AgentChatAvailability = 'available' | 'unavailable' | 'unknown' | 'no_access' | 'in_use'

export interface DirectAgentRow {
  /** Source-qualified identity — the ONLY thing a selection may bind to. */
  connectionId: null | string
  profile: string
  /** Backend profile name the RPC must actually address; can diverge from
   *  `profile` for an aliased/remote row. Blank when the roster entry
   *  omitted its own backend identity — such a row is never `available`
   *  (see `deriveDirectAgentRoster`) and must never be handed to
   *  `host.openCanonicalAgentChat`, which requires a non-blank explicit
   *  value and rejects a substituted one (Architect corrective,
   *  2026-09-02, seventh pass). */
  targetProfile: string
  /** Roster-reported display identity, presentation only — never the
   *  selection key. */
  handle: string
  connectionLabel: string
  availability: AgentChatAvailability
  reason: string
}
