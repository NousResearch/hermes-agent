/**
 * Direct agent roster derivation (Mode 2 entry, Two Owner Communication
 * Modes release §3.1/§4). Pure — no host access — so eligibility is
 * testable in isolation. `page.tsx`'s roster screen calls this against the
 * live `host.agents()` result; nothing here reaches for a hardcoded list,
 * a filesystem scan, or a guessed profile.
 */

import type { AgentChatAvailability, DirectAgentRow } from './agent-chat-types'

interface RosterAgentInput {
  connectionId?: null | string
  connectionKind?: string
  connectionLabel?: string
  profile: string
  targetProfile?: string
  handle?: string
}

interface RosterSourceInput {
  connectionId: string
  kind: string
  label: string
  reachable: boolean
  error?: string
}

export interface AgentRosterResultInput {
  agents: RosterAgentInput[]
  sources: RosterSourceInput[]
}

/** Every eligible-selection row comes off `host.agents()` — never a
 *  hardcoded roster. Sources with an enumeration error still produce a row
 *  (marked unavailable with the real reason), rather than silently
 *  disappearing — the release requires visible unavailable/no-access
 *  states, not omission. */
export function deriveDirectAgentRoster(input: AgentRosterResultInput): DirectAgentRow[] {
  const seen = new Set<string>()
  const rows: DirectAgentRow[] = []

  for (const agent of input.agents ?? []) {
    const connectionId = (agent.connectionId ?? '').trim() || null
    const profile = (agent.profile ?? '').trim()

    if (!profile) {
      continue
    }

    const key = `${connectionId ?? ''}\u0000${profile}`

    if (seen.has(key)) {
      continue
    }

    seen.add(key)

    const targetProfile = (agent.targetProfile ?? '').trim()
    const handle = (agent.handle ?? '').trim() || profile
    const connectionLabel = (agent.connectionLabel ?? '').trim() || 'This device'

    // A roster row missing/blank targetProfile can never be handed to
    // host.openCanonicalAgentChat's exact {connectionId, profile,
    // targetProfile} admission — that seam requires a non-blank explicit
    // backend identity and rejects a substituted one (Architect
    // corrective, 2026-09-02, seventh pass: this roster derivation
    // previously defaulted a missing targetProfile to `profile`, which
    // could authorize a target the roster never actually vouched for).
    // Surface the row as unavailable rather than silently omitting it or
    // rendering it as a working, selectable card.
    if (!targetProfile) {
      rows.push({
        connectionId,
        profile,
        targetProfile: '',
        handle,
        connectionLabel,
        availability: 'unavailable',
        reason: 'Appears in the roster, but the runtime did not report a complete backend identity for this agent.',
      })

      continue
    }

    rows.push({
      connectionId,
      profile,
      targetProfile,
      handle,
      connectionLabel,
      availability: 'available',
      reason: 'Confirmed reachable via the current authorized runtime roster.',
    })
  }

  for (const source of input.sources ?? []) {
    if (source.reachable) {
      continue
    }

    const connectionId = (source.connectionId ?? '').trim() || null
    const key = `${connectionId ?? ''}\u0000`

    if (seen.has(key)) {
      continue
    }

    seen.add(key)

    const availability: AgentChatAvailability = 'unavailable'
    const detail = (source.error ?? '').trim()

    rows.push({
      connectionId,
      profile: '',
      targetProfile: '',
      handle: (source.label ?? '').trim() || 'Unavailable source',
      connectionLabel: (source.label ?? '').trim() || 'Unavailable source',
      availability,
      reason: detail
        ? `Appears in the roster, but the connection could not be reached (${detail}).`
        : 'Appears in the roster, but the connection could not be reached.',
    })
  }

  return rows
}
