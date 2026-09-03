/**
 * Command Center — Two Owner Communication Modes UI components.
 *
 * Mode 2 (direct agent chat): a roster-first selection flow, then the
 * per-agent handoff card, both delegating the actual conversation to the
 * existing Hermes Desktop Bot Mode canonical-chat seam
 * (`host.openCanonicalAgentChat`) — never a Command-Center-owned transcript.
 *
 * Mode 1 (Underboss orchestration): a single handoff entry plus a truthful
 * "Orchestration trace unavailable" gate-note. No fabricated trace, ever.
 *
 * Visual source of truth: Picasso's approved mockups
 * `screen-agent-roster.html`, `screen-underboss-thread.html`, and the
 * updated `agent-view/index.html` (Architect disposition, 2026-09-01).
 */

import { host } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { type AgentRosterResultInput, deriveDirectAgentRoster } from './agent-chat-roster'
import type { AgentChatAvailability, DirectAgentRow } from './agent-chat-types'
import { GateNote } from './components'

// ── Shared: current authorized roster, re-fetched on every mount ───────────

type RosterState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; rows: DirectAgentRow[] }

/** Loads the current authorized runtime roster via `host.agents()` and
 *  derives eligible direct-chat rows from it. Never caches across mounts —
 *  each screen visit re-checks eligibility fresh, per the release's
 *  "current authorized roster" requirement. */
function useDirectAgentRoster(): RosterState {
  const [state, setState] = useState<RosterState>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false

    setState({ status: 'loading' })

    void (async () => {
      try {
        const result = (await host.agents()) as unknown as AgentRosterResultInput
        const rows = deriveDirectAgentRoster(result)

        if (!cancelled) {
          setState({ status: 'ready', rows })
        }
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : String(error || '')

          setState({ status: 'error', message: message || 'Could not load the agent roster.' })
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [])

  return state
}

// ── Shared: open-conversation action ────────────────────────────────────────

export type OpenConversationStatus =
  | { kind: 'idle' }
  | { kind: 'opening' }
  | { kind: 'error'; message: string }
  | { kind: 'in_use' }

/** The ONLY way this plugin ever reaches a real agent conversation: the
 *  existing Bot Mode canonical-chat seam via `host.openCanonicalAgentChat`.
 *  No fallback to a guessed profile, no second chat surface. A live-session
 *  ownership conflict renders as "in use", never bypassed. */
export async function openAgentConversation(
  target: { connectionId: null | string; profile: string; targetProfile: string },
  setStatus: (status: OpenConversationStatus) => void
): Promise<void> {
  setStatus({ kind: 'opening' })

  try {
    const opened = await host.openCanonicalAgentChat(target)

    if (!opened) {
      setStatus({ kind: 'error', message: 'Could not open this conversation.' })

      return
    }

    setStatus({ kind: 'idle' })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error || '')

    if (/already has a live owner|in use/i.test(message)) {
      setStatus({ kind: 'in_use' })

      return
    }

    setStatus({ kind: 'error', message: message || 'Could not open this conversation.' })
  }
}

const AVAILABILITY_LABEL: Record<AgentChatAvailability, string> = {
  available: 'Available',
  unavailable: 'Unavailable',
  unknown: 'Unknown',
  no_access: 'No access',
  in_use: 'In use — open in Hermes',
}

const AVAILABILITY_CLASS: Record<AgentChatAvailability, string> = {
  available: 'available',
  unavailable: 'unavailable',
  unknown: 'unknown',
  no_access: 'no-access',
  in_use: 'in-use',
}

// ── Mode 2: Direct Agent Roster ─────────────────────────────────────────────

interface DirectAgentRosterScreenProps {
  onSelectAgent: (row: DirectAgentRow) => void
}

export function DirectAgentRosterScreen({ onSelectAgent }: DirectAgentRosterScreenProps) {
  const roster = useDirectAgentRoster()

  const available = roster.status === 'ready' ? roster.rows.filter(r => r.availability === 'available') : []
  const notReachable = roster.status === 'ready' ? roster.rows.filter(r => r.availability !== 'available') : []

  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Agents · Direct communication</div>
        <h1 className="cc-h1">Talk to an individual agent directly.</h1>
        <div className="cc-subhead">
          Selecting an agent opens that agent&apos;s existing conversation in Hermes — the same one you&apos;d reach
          directly. Command Center does not own, store, or duplicate that conversation. A message to an agent is not
          a task, approval, or delegation on its own.
        </div>
      </div>
      <div className="cc-page-content">
        <GateNote>
          Agent Roster → eligible-selection check against the current authorized Hermes runtime roster → existing
          profile/route → that agent&apos;s own canonical Bot Chat, opened in Hermes. A roster entry that cannot be
          verified renders as Unavailable, No Access, or Unknown, never as a working card.
        </GateNote>

        {roster.status === 'loading' && (
          <div style={{ padding: '24px 0', fontFamily: 'var(--cc-font-mono)', fontSize: 11, color: 'var(--cc-text-faint)' }}>
            Loading current roster…
          </div>
        )}
        {roster.status === 'error' && (
          <div style={{ padding: '24px 0', fontFamily: 'var(--cc-font-mono)', fontSize: 11, color: 'var(--cc-red)' }}>
            {roster.message}
          </div>
        )}

        {roster.status === 'ready' && (
          <>
            <div className="cc-section-label">Eligible for direct conversation ({available.length})</div>
            <div className="cc-grid-2" style={{ gap: 14, marginBottom: 28 }}>
              {available.map(row => (
                <DirectAgentCard key={`${row.connectionId ?? ''}::${row.profile}`} onSelect={() => onSelectAgent(row)} row={row} />
              ))}
              {available.length === 0 && (
                <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 11, color: 'var(--cc-text-faint)' }}>
                  No agents are currently eligible for direct conversation.
                </div>
              )}
            </div>

            {notReachable.length > 0 && (
              <>
                <div className="cc-section-label">Not currently reachable ({notReachable.length})</div>
                <div className="cc-grid-2" style={{ gap: 14 }}>
                  {notReachable.map(row => (
                    <DirectAgentCard key={`${row.connectionId ?? ''}::${row.profile}`} row={row} />
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function DirectAgentCard({ row, onSelect }: { row: DirectAgentRow; onSelect?: () => void }) {
  const selectable = row.availability === 'available' && Boolean(onSelect)

  return (
    <div
      className="cc-card"
      onClick={selectable ? onSelect : undefined}
      role={selectable ? 'button' : undefined}
      style={{ display: 'flex', gap: 14, alignItems: 'flex-start', cursor: selectable ? 'pointer' : 'not-allowed', opacity: selectable ? 1 : 0.72 }}
      tabIndex={selectable ? 0 : undefined}
    >
      <div aria-hidden="true" className="cc-avatar-xl" />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 14, fontWeight: 600 }}>{row.handle}</div>
        <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: '9.5px', color: 'var(--cc-text-faint)', textTransform: 'uppercase', letterSpacing: '0.4px', marginTop: 2 }}>
          {row.connectionLabel}
        </div>
        <div style={{ fontSize: 12, color: 'var(--cc-text-dim)', marginTop: 8, lineHeight: 1.45 }}>{row.reason}</div>
        <span className={`cc-roster-chip ${AVAILABILITY_CLASS[row.availability]}`}>{AVAILABILITY_LABEL[row.availability]}</span>
      </div>
    </div>
  )
}

// ── Mode 2: Direct Agent handoff card (embedded in Agent View) ─────────────

export function DirectAgentHandoffCard({ row }: { row: DirectAgentRow }) {
  const [status, setStatus] = useState<OpenConversationStatus>({ kind: 'idle' })

  return (
    <div style={{ marginBottom: 20 }}>
      <button
        className="cc-btn primary"
        disabled={status.kind === 'opening'}
        onClick={() => void openAgentConversation({ connectionId: row.connectionId, profile: row.profile, targetProfile: row.targetProfile }, setStatus)}
        type="button"
      >
        {status.kind === 'opening' ? 'Opening…' : `Open ${row.handle}'s Hermes conversation →`}
      </button>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--cc-text-faint)', lineHeight: 1.5, maxWidth: 480 }}>
        Opens the real, native Hermes conversation. Command Center does not own or store this transcript. Sending a
        message there is not a Command Center action and does not itself create task, approval, or delegation
        authority.
      </div>
      {status.kind === 'in_use' && (
        <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--cc-violet)' }}>
          This conversation is already open elsewhere — open it directly in Hermes instead of here.
        </div>
      )}
      {status.kind === 'error' && (
        <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--cc-red)' }}>{status.message}</div>
      )}
    </div>
  )
}

// ── Mode 1: Underboss orchestration entry ───────────────────────────────────

/** Underboss is identified by profile name from the current authorized
 *  roster — same eligibility rule as Mode 2, never a hardcoded assumption
 *  that the row exists. */
function findUnderbossRow(roster: RosterState): DirectAgentRow | null {
  if (roster.status !== 'ready') {
    return null
  }

  return roster.rows.find(row => row.availability === 'available' && row.profile.toLowerCase() === 'underboss') ?? null
}

/** Resolves whether a given M1 agent-directory id has a matching row in the
 *  CURRENT authorized runtime roster. The M1 synthetic Agents grid and the
 *  real Bot Mode roster are two independent identities — Agent View must
 *  never assume they match. Only when the live roster itself confirms
 *  availability does the handoff CTA render. */
export function useAgentChatEligibility(candidateProfile: string): DirectAgentRow | null {
  const roster = useDirectAgentRoster()

  if (roster.status !== 'ready' || !candidateProfile) {
    return null
  }

  return (
    roster.rows.find(
      row => row.availability === 'available' && row.profile.toLowerCase() === candidateProfile.toLowerCase()
    ) ?? null
  )
}

export function UnderbossEntryScreen() {
  const roster = useDirectAgentRoster()
  const underboss = findUnderbossRow(roster)
  const [status, setStatus] = useState<OpenConversationStatus>({ kind: 'idle' })

  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Agents / Underboss · Global orchestration entry</div>
        <h1 className="cc-h1">Underboss</h1>
        <div className="cc-subhead">Executive coordination · orchestration &amp; governance</div>
      </div>
      <div className="cc-page-content">
        <div className="cc-section-label">Talk to Underboss</div>
        <div className="cc-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, padding: 22 }}>
          <div style={{ fontSize: 13, color: 'var(--cc-text-dim)', lineHeight: 1.55, maxWidth: 420 }}>
            Underboss&apos;s actual conversation — history, composer, and every reply — lives in Hermes. This opens
            it directly; <strong style={{ color: 'var(--cc-text)' }}>Command Center does not embed or duplicate it.</strong>
          </div>
          <button
            className="cc-btn primary"
            disabled={!underboss || status.kind === 'opening'}
            onClick={() =>
              underboss && void openAgentConversation({ connectionId: underboss.connectionId, profile: underboss.profile, targetProfile: underboss.targetProfile }, setStatus)
            }
            type="button"
          >
            {status.kind === 'opening' ? 'Opening…' : 'Open Underboss\u2019s Hermes conversation →'}
          </button>
        </div>
        {roster.status === 'ready' && !underboss && (
          <div style={{ marginTop: 10, fontSize: 11.5, color: 'var(--cc-text-faint)' }}>
            Underboss is not present in the current authorized runtime roster — this handoff is unavailable until it
            is.
          </div>
        )}
        {roster.status === 'error' && (
          <div style={{ marginTop: 10, fontSize: 11.5, color: 'var(--cc-red)' }}>{roster.message}</div>
        )}
        {status.kind === 'in_use' && (
          <div style={{ marginTop: 10, fontSize: 11.5, color: 'var(--cc-violet)' }}>
            This conversation is already open elsewhere — open it directly in Hermes instead of here.
          </div>
        )}
        {status.kind === 'error' && (
          <div style={{ marginTop: 10, fontSize: 11.5, color: 'var(--cc-red)' }}>{status.message}</div>
        )}
        <GateNote>
          This opens Underboss&apos;s existing canonical Hermes Bot Chat through the existing Desktop
          session-handoff seam. Command Center does not dispatch, delegate, relay, or persist any message on your
          behalf — sending happens only inside the Hermes conversation itself, after you type and submit it there.
        </GateNote>

        <div className="cc-section-label" style={{ marginTop: 28 }}>Resulting orchestration</div>
        <div className="cc-gate-note" style={{ display: 'flex', gap: 14, alignItems: 'flex-start', padding: '20px 22px' }}>
          <span style={{ fontSize: 18, color: 'var(--cc-amber)', flexShrink: 0 }}>⚠</span>
          <div>
            <div style={{ fontSize: '13.5px', fontWeight: 600, marginBottom: 6 }}>Orchestration trace unavailable</div>
            <div style={{ fontSize: 12, color: 'var(--cc-text-dim)', lineHeight: 1.6 }}>
              Command Center cannot yet show which agents were involved, what was delegated, or how your request was
              handled downstream. Hermes does not currently provide one authoritative source connecting an Owner
              request to Underboss&apos;s routing, delegation, and worker activity — so no trace is shown, rather
              than an inferred or guessed one. This instruments in a future, separately-approved release once a
              verified correlation source exists.
            </div>
            <span className="cc-chip unavailable" style={{ marginTop: 10, display: 'inline-block' }}>Not yet instrumented</span>
          </div>
        </div>
      </div>
    </div>
  )
}
