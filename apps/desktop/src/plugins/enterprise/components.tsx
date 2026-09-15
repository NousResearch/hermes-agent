/**
 * Command Center M1 — shared components.
 *
 * Implements §2 shared component contracts exactly once.
 * All screens import from here — no per-screen rebuilds.
 */

import { type ReactNode } from 'react'

import type {
  AttentionItem,
  ConfidenceLevel,
  Envelope,
  ImplementationStatus,
  LiveStatus,
  ObjectiveSummary,
} from './types'

// ── Synthetic data badge ────────────────────────────────────────────────────
export function SyntheticBadge() {
  return (
    <span aria-label="Simulated test data — not live Hermes output" className="cc-synthetic-banner">
      ◈ SIMULATED · M1
    </span>
  )
}

// ── Gate note (reusable blocked-dependency pattern) ─────────────────────────
export function GateNote({ children }: { children: ReactNode }) {
  return <div className="cc-gate-note">⚠ {children}</div>
}

// ── Navigation ──────────────────────────────────────────────────────────────
export type Screen =
  | 'home' | 'goals' | 'agents' | 'orchestration' | 'knowledge'
  | 'orgmap' | 'history' | 'health' | 'briefings' | 'security'

const NAV_ITEMS: Array<{ id: Screen; label: string }> = [
  { id: 'home',          label: 'Home' },
  { id: 'goals',         label: 'Goals' },
  { id: 'agents',        label: 'Agents' },
  { id: 'orchestration', label: 'Orchestration' },
  { id: 'knowledge',     label: 'Knowledge' },
  { id: 'orgmap',        label: 'Org Map' },
  { id: 'history',       label: 'History' },
  { id: 'health',        label: 'System Health' },
  { id: 'briefings',     label: 'Briefings' },
  { id: 'security',      label: 'Security' },
]

interface TopBarProps {
  active: Screen
  onNavigate: (s: Screen) => void
  decisionCount?: number
  actionCount?: number
}

export function TopBar({ active, onNavigate, decisionCount = 0, actionCount = 0 }: TopBarProps) {
  const dotColor = (id: Screen) => {
    if (id === 'home') {return decisionCount + actionCount > 0 ? 'var(--cc-amber)' : 'var(--cc-text-faint)'}

    if (id === 'health') {return 'var(--cc-amber)'}

    return 'var(--cc-text-faint)'
  }

  return (
    <div aria-label="Command Center navigation" className="cc-topbar" role="navigation">
      <div className="cc-brand">
        <div aria-hidden="true" className="cc-brand-mark" />
        <span className="cc-brand-name">Command</span>
      </div>
      {NAV_ITEMS.map(item => (
        <button
          aria-current={active === item.id ? 'page' : undefined}
          className={`cc-station${active === item.id ? ' active' : ''}`}
          key={item.id}
          onClick={() => onNavigate(item.id)}
          type="button"
        >
          <span aria-hidden="true" className="cc-station-dot" style={{ background: dotColor(item.id) }} />
          {item.label}
          {item.id === 'home' && decisionCount > 0 && (
            <span aria-label={`${decisionCount} decisions need attention`} className="cc-station-badge">
              {decisionCount}
            </span>
          )}
        </button>
      ))}
      <div className="cc-topbar-right">
        <SyntheticBadge />
        <div aria-hidden="true" className="cc-cmd-hint">⌘K&nbsp; Command…</div>
      </div>
    </div>
  )
}

// ── Attention item card (§2.1) ───────────────────────────────────────────────
interface AttentionCardProps {
  item: AttentionItem
  onAction?: (item: AttentionItem) => void
}

function relativeTime(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime()
  const h = Math.floor(ms / 3_600_000)

  if (h < 1) {return 'Just now'}

  if (h < 24) {return `${h}h ago`}
  const d = Math.floor(h / 24)

  if (d === 1) {return 'Yesterday'}

  return `${d}d ago`
}

export function AttentionCard({ item, onAction }: AttentionCardProps) {
  return (
    <div className={`cc-item ${item.classification}`} role="article">
      <div className="cc-item-body">
        <div className="cc-item-top">
          <span className={`cc-tag ${item.classification}`}>
            {item.classification === 'decision' ? 'Decision required'
              : item.classification === 'action' ? 'Action required'
              : 'Information'}
          </span>
          <span className="cc-item-time">{relativeTime(item.raised_at)}</span>
        </div>
        <div className="cc-item-title">{item.headline}</div>
        <div className="cc-item-desc">{item.description}</div>
        <div className="cc-item-meta">
          <span className="cc-agent-chip">
            <span aria-hidden="true" className="cc-avatar" />
            {item.raised_by.agent_name}
          </span>
          {item.evidence_confidence && item.recommendation_confidence && (
            <span style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 10, color: 'var(--cc-text-faint)', marginLeft: 'auto', display: 'flex', gap: 10 }}>
              Evidence <strong style={{ color: 'var(--cc-text)' }}>{capitalize(item.evidence_confidence)}</strong>
              {' · '}Confidence <strong style={{ color: 'var(--cc-text)' }}>{capitalize(item.recommendation_confidence)}</strong>
            </span>
          )}
        </div>
      </div>
      <div className="cc-item-actions">
        <button
          aria-label={`${item.primary_action_label}: ${item.headline}`}
          className="cc-btn primary"
          disabled={!onAction}  // Decisions/Actions open the Decision flow; Information items have nothing to do (presentation only in M1)
          onClick={() => onAction?.(item)}
          type="button"
        >
          {item.primary_action_label}
        </button>
      </div>
    </div>
  )
}

function capitalize(s: string) { return s.charAt(0).toUpperCase() + s.slice(1) }

// ── Status chip (§2.2) ───────────────────────────────────────────────────────
type ChipVariant =
  | 'live' | 'proposed' | 'unavailable' | 'canonical' | 'superseded'
  | 'unresolved' | 'draft' | 'simulated' | 'stale'

export function StatusChip({ variant, label }: { variant: ChipVariant; label?: string }) {
  const defaults: Record<ChipVariant, string> = {
    live: 'Live',
    proposed: 'Proposed',
    unavailable: 'Unavailable',
    canonical: 'Canonical',
    superseded: 'Superseded',
    unresolved: 'Unresolved',
    draft: 'Draft',
    simulated: 'Simulated',
    stale: 'Stale',
  }

  return <span className={`cc-chip ${variant}`}>{label ?? defaults[variant]}</span>
}

// ── Agent identity chip (§2.5) ───────────────────────────────────────────────
const STATUS_DOT_COLOR: Record<string, string> = {
  running: 'var(--cc-green)',
  idle: 'var(--cc-text-faint)',
  blocked: 'var(--cc-amber)',
  unavailable: 'var(--cc-red)',
}

export function AgentCard({
  name, role, focus, implementationStatus, liveStatus, onClick,
}: {
  name: string
  role: string
  focus: string | null
  implementationStatus: ImplementationStatus
  liveStatus: LiveStatus
  onClick?: () => void
}) {
  const chipLabel = implementationStatus === 'proposed' ? 'Proposed · Unbuilt' : 'Live · Owned'
  const chipVariant: ChipVariant = implementationStatus === 'proposed' ? 'proposed' : 'live'

  return (
    <div
      className="cc-card"
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      style={{ display: 'flex', gap: 14, alignItems: 'flex-start', cursor: onClick ? 'pointer' : undefined }}
      tabIndex={onClick ? 0 : undefined}
    >
      <div aria-hidden="true" className="cc-avatar-xl">
        {liveStatus && (
          <span
            aria-hidden="true"
            className="cc-status-dot"
            style={{ background: STATUS_DOT_COLOR[liveStatus] ?? 'var(--cc-text-faint)' }}
          />
        )}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 14, fontWeight: 600 }}>{name}</div>
        <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: '9.5px', color: 'var(--cc-text-faint)', textTransform: 'uppercase', letterSpacing: '0.4px', marginTop: 2 }}>
          {role}
        </div>
        {focus && (
          <div style={{ fontSize: 12, color: 'var(--cc-text-dim)', marginTop: 8, lineHeight: 1.4 }}>{focus}</div>
        )}
        <div style={{ marginTop: 10 }}>
          <StatusChip label={chipLabel} variant={chipVariant} />
        </div>
      </div>
    </div>
  )
}

// ── Confidence pair (§2.3) ───────────────────────────────────────────────────
export function ConfidencePair({
  evidenceLevel, evidenceExplain, recommendationLevel, recommendationExplain,
}: {
  evidenceLevel: ConfidenceLevel
  evidenceExplain: string
  recommendationLevel: ConfidenceLevel
  recommendationExplain: string
}) {
  return (
    <div className="cc-conf-row">
      <div className="cc-conf-box">
        <div className="cc-conf-label">Evidence confidence</div>
        <div className={`cc-conf-val ${evidenceLevel}`}>{capitalize(evidenceLevel)}</div>
        <div className="cc-conf-explain">{evidenceExplain}</div>
      </div>
      <div className="cc-conf-box">
        <div className="cc-conf-label">Recommendation confidence</div>
        <div className={`cc-conf-val ${recommendationLevel}`}>{capitalize(recommendationLevel)}</div>
        <div className="cc-conf-explain">{recommendationExplain}</div>
      </div>
    </div>
  )
}

// ── Objective / goal row ─────────────────────────────────────────────────────
export function ObjectiveRow({ obj }: { obj: ObjectiveSummary }) {
  const color = obj.status === 'at_risk' ? 'var(--cc-red)'
    : obj.status === 'on_track' ? 'var(--cc-green)'
    : 'var(--cc-text-faint)'

  const label = obj.status === 'at_risk' ? 'At risk'
    : obj.status === 'on_track' ? 'On track'
    : obj.status === 'blocked' ? 'Blocked' : 'Done'

  const barClass = obj.status === 'on_track' ? 'green' : 'amber'

  return (
    <div style={{ padding: '11px 0', borderBottom: '1px solid var(--cc-border-soft)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 7 }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>{obj.name}</span>
        <span style={{ fontFamily: 'var(--cc-font-mono)', fontSize: '9.5px', textTransform: 'uppercase', letterSpacing: '0.5px', color }}>{label}</span>
      </div>
      <div className="cc-bar">
        <div className={`cc-bar-fill ${barClass}`} style={{ width: `${obj.progress_pct}%` }} />
      </div>
      <div style={{ fontSize: 11, color: 'var(--cc-text-faint)', marginTop: 6 }}>{obj.note}</div>
    </div>
  )
}

// ── Loading / unavailable placeholder ───────────────────────────────────────
export function Placeholder({ label, state = 'loading' }: { label?: string; state?: 'loading' | 'unavailable' | 'error' }) {
  const msg = state === 'loading' ? (label ?? 'Loading…')
    : state === 'unavailable' ? (label ?? 'Unavailable')
    : (label ?? 'Error loading data')

  const color = state === 'loading' ? 'var(--cc-text-faint)' : state === 'error' ? 'var(--cc-red)' : 'var(--cc-text-faint)'

  return (
    <div style={{ padding: '24px 0', fontFamily: 'var(--cc-font-mono)', fontSize: 11, color, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
      {msg}
    </div>
  )
}

// ── Envelope unwrap helper (type-safe) ───────────────────────────────────────
export function unwrap<T>(env: Envelope<T> | undefined): { data: T | null; status: string; isSynthetic: boolean } {
  if (!env) {return { data: null, status: 'loading', isSynthetic: false }}

  return {
    data: env.data,
    status: env.status,
    isSynthetic: env.source === 'synthetic-m1',
  }
}
