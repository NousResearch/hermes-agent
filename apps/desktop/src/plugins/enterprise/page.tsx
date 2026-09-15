/**
 * Command Center M1 — Owner Command Center page.
 *
 * Implements all 12 Picasso-approved screens:
 *   Home, Goals, Agents, Orchestration, Knowledge, Org Map,
 *   History, System Health, Briefings, Security, Agent View,
 *   Decision Approval (consequential action flow)
 *
 * Visual source of truth: Picasso's BUILDER_SPEC_v1.0.md handoff and its 12
 * approved desktop mockups (external design artifacts, not part of this repo).
 *
 * All data is from the synthetic M1 provider. Source="synthetic-m1".
 * No real Hermes, Syntex, EKV, or external service is contacted.
 *
 * G3–G7 compliance:
 *   - Command execution: disabled (all action buttons present but disabled in M1)
 *   - Orb: neutral UNKNOWN only; no health inference
 *   - Knowledge: gate-note banner; no Vault/EKV access
 *   - Security: display-only; no remediation authority
 *   - Synthetic data never presented as live
 */

import { useQuery } from '@hermes/plugin-sdk'
import { Fragment, useEffect, useRef, useState } from 'react'

import { DirectAgentHandoffCard, DirectAgentRosterScreen, UnderbossEntryScreen, useAgentChatEligibility } from './agent-chat-components'
import type { DirectAgentRow } from './agent-chat-types'
import {
  ccKeys,
  fetchAgents,
  fetchOrbState,
  fetchSecurity,
  fetchSummary,
  fetchWork,
  POLL_AGENTS_MS,
  POLL_ORB_MS,
  POLL_SECURITY_MS,
  POLL_SUMMARY_MS,
  POLL_WORK_MS,
} from './api'
import {
  AgentCard,
  AttentionCard,
  ConfidencePair,
  GateNote,
  ObjectiveRow,
  Placeholder,
  type Screen,
  StatusChip,
  TopBar,
  unwrap,
} from './components'
import type { AttentionItem } from './types'

// ── Decision Approval (§2.4) — 5-state flow, presentation only in M1 ────────
type ApprovalState = 'proposed' | 'modify' | 'consequence' | 'approve' | 'execute'

function DecisionApprovalScreen({ item, onClose }: { item: AttentionItem; onClose: () => void }) {
  const [state, setState] = useState<ApprovalState>('proposed')
  const [hasModified, setHasModified] = useState(false)
  const [instructionText, setInstructionText] = useState('')
  const [removed, setRemoved] = useState(false)

  const steps: ApprovalState[] = ['proposed', 'modify', 'consequence', 'approve', 'execute']

  const stepLabels: Record<ApprovalState, string> = {
    proposed: '① Proposed',
    modify: '② Modify',
    consequence: '③ Consequences',
    approve: '④ Approve',
    execute: '⑤ Execute',
  }

  const canGoToConsequence = state === 'modify' && hasModified
  const canApprove = state === 'consequence' || (state === 'modify' && !hasModified)

  return (
    <div style={{ background: 'var(--cc-bg)', minHeight: '100%', display: 'flex', justifyContent: 'center', padding: '40px 20px', overflowY: 'auto' }}>
      <div style={{ width: '100%', maxWidth: 760 }}>
        {/* Step tracker */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 26, fontFamily: 'var(--cc-font-mono)', fontSize: 10, letterSpacing: '0.5px', textTransform: 'uppercase', color: 'var(--cc-text-faint)' }}>
          {steps.map((s, i) => (
            <Fragment key={s}>
              <span style={{ color: s === state ? 'var(--cc-amber)' : steps.indexOf(s) < steps.indexOf(state) ? 'var(--cc-green)' : undefined, fontWeight: s === state ? 600 : undefined }}>
                {stepLabels[s]}
              </span>
              {i < steps.length - 1 && <span style={{ width: 16, height: 1, background: 'var(--cc-border)', display: 'inline-block' }} />}
            </Fragment>
          ))}
        </div>

        {/* Main card */}
        <div className="cc-card cc-card-lg" style={{ marginBottom: 16 }}>
          <div className="cc-eyebrow">Decision required · Raised by {item.raised_by.agent_name}</div>
          <h2 style={{ fontFamily: 'var(--cc-font-display)', fontWeight: 600, fontSize: 24, lineHeight: 1.25, marginBottom: 8 }}>{item.headline}</h2>
          <div style={{ fontSize: 13, color: 'var(--cc-text-dim)', lineHeight: 1.6, maxWidth: 600 }}>{item.description}</div>
          {item.evidence_confidence && item.recommendation_confidence && (
            <ConfidencePair
              evidenceExplain="Based on completed investigation, not projection."
              evidenceLevel={item.evidence_confidence}
              recommendationExplain="Agents reached different conclusions from the same evidence."
              recommendationLevel={item.recommendation_confidence}
            />
          )}
        </div>

        {/* Modify card */}
        {(state === 'modify' || state === 'consequence') && (
          <div className="cc-card cc-card-lg" style={{ marginBottom: 16 }}>
            <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: '10.5px', letterSpacing: '1px', textTransform: 'uppercase', color: 'var(--cc-text-faint)', margin: '0 0 12px' }}>
              Who's involved in resolving this
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px', border: '1px solid var(--cc-border-soft)', borderRadius: 9, marginBottom: 8, background: 'rgba(255,255,255,0.015)' }}>
              <div aria-hidden="true" className="cc-avatar-lg" />
              <div>
                <div style={{ fontSize: '12.5px', fontWeight: 600 }}>{item.raised_by.agent_name}</div>
                <div style={{ fontSize: '10.5px', color: 'var(--cc-text-faint)' }}>Owns this decision</div>
              </div>
              <span style={{ marginLeft: 'auto', fontFamily: 'var(--cc-font-mono)', fontSize: '9.5px', color: 'var(--cc-text-faint)', border: '1px solid var(--cc-border)', padding: '3px 8px', borderRadius: 5, opacity: 0.5, cursor: 'not-allowed' }} title="Command execution disabled in M1">
                Adjust scope
              </span>
            </div>
            {!removed && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px', border: '1px solid var(--cc-border-soft)', borderRadius: 9, marginBottom: 8, background: 'rgba(255,255,255,0.015)' }}>
                <div aria-hidden="true" className="cc-avatar-lg" />
                <div>
                  <div style={{ fontSize: '12.5px', fontWeight: 600 }}>Growth Intelligence Manager</div>
                  <div style={{ fontSize: '10.5px', color: 'var(--cc-text-faint)' }}>Provided the counter-recommendation</div>
                </div>
                <button
                  onClick={() => { setRemoved(true); setHasModified(true) }}
                  style={{ marginLeft: 'auto', fontFamily: 'var(--cc-font-mono)', fontSize: '9.5px', color: 'var(--cc-red)', border: '1px solid rgba(226,96,79,0.3)', padding: '3px 8px', borderRadius: 5, cursor: 'pointer', background: 'none' }}
                  type="button"
                >
                  Remove
                </button>
              </div>
            )}
            <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: '10.5px', letterSpacing: '1px', textTransform: 'uppercase', color: 'var(--cc-text-faint)', margin: '22px 0 12px' }}>
              Your instructions (optional)
            </div>
            <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--cc-border-soft)', borderRadius: 9, padding: '12px 14px' }}>
              <textarea
                onChange={e => { setInstructionText(e.target.value);

 if (e.target.value) {setHasModified(true)} }}
                placeholder="Add context or constraints before approving…"
                rows={2}
                style={{ all: 'unset' as any, width: '100%', display: 'block', color: 'var(--cc-text)', fontSize: '12.5px', lineHeight: 1.55, fontFamily: 'var(--cc-font-body)', resize: 'vertical' }}
                value={instructionText}
              />
            </div>
          </div>
        )}

        {/* Consequence preview — mandatory if modify produced a change */}
        {removed && (state === 'consequence' || state === 'approve') && (
          <div className="cc-card" style={{ marginBottom: 16 }}>
            <div style={{ background: 'linear-gradient(135deg, var(--cc-amber-glow), transparent 60%), var(--cc-bg-card)', border: '1px solid var(--cc-border)', borderLeft: '3px solid var(--cc-amber)', borderRadius: 12, padding: '16px 18px' }}>
              <span style={{ fontFamily: 'var(--cc-font-mono)', fontSize: '9.5px', letterSpacing: '0.6px', textTransform: 'uppercase', color: 'var(--cc-amber)', marginBottom: 8, display: 'block' }}>
                Consequence of removing Growth Intelligence Manager
              </span>
              <ul style={{ margin: '8px 0 0 18px', fontSize: '12.5px', color: 'var(--cc-text-dim)', lineHeight: 1.7 }}>
                <li><strong style={{ color: 'var(--cc-text)' }}>Recommendation confidence will not update</strong> — Growth Intelligence's position is on record but they won't be consulted on follow-up refinement.</li>
                <li><strong style={{ color: 'var(--cc-text)' }}>{item.raised_by.agent_name} becomes sole owner</strong> of finalizing the scope decision.</li>
                <li>This does not change {item.raised_by.agent_name}'s existing authority — it only narrows who's actively consulted.</li>
              </ul>
            </div>
          </div>
        )}

        {/* M1 disabled notice */}
        <GateNote>
          Command execution is disabled in M1. This flow demonstrates the full 5-state consequential-action UX but does not submit, notify, or log any real action. Approve/Execute buttons are present but no effect occurs.
        </GateNote>

        {/* Actions */}
        <div style={{ display: 'flex', gap: 10, marginTop: 22 }}>
          <button className="cc-btn ghost" onClick={onClose} type="button">Cancel</button>
          {state === 'proposed' && (
            <>
              <button className="cc-btn ghost" onClick={() => setState('modify')} type="button">Modify plan</button>
              <button className="cc-btn primary" disabled onClick={() => setState('approve')} style={{ flex: 1 }} type="button">
                Approve as proposed (disabled in M1)
              </button>
            </>
          )}
          {state === 'modify' && !hasModified && (
            <button className="cc-btn primary" disabled onClick={() => setState('approve')} style={{ flex: 1 }} type="button">
              Approve as proposed (disabled in M1)
            </button>
          )}
          {state === 'modify' && hasModified && (
            <button className="cc-btn primary" onClick={() => setState('consequence')} style={{ flex: 1 }} type="button">
              Preview consequences
            </button>
          )}
          {state === 'consequence' && (
            <button className="cc-btn primary" disabled onClick={() => setState('approve')} style={{ flex: 1 }} type="button">
              Approve with these changes (disabled in M1)
            </button>
          )}
          {state === 'approve' && (
            <button className="cc-btn primary" disabled style={{ flex: 1 }} type="button">
              Execute (disabled in M1)
            </button>
          )}
        </div>
        <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 10, color: 'var(--cc-text-faint)', textAlign: 'center', marginTop: 14 }}>
          In production: approving would notify affected agents and log this decision to History with full provenance.
        </div>
      </div>
    </div>
  )
}

// ── Constellation (right-side hero atmosphere — org-as-connected-nodes) ──────
// Static/decorative per the locked a-hero/index.html spec (§ "right-side
// constellation — organization as connected nodes"). Illustrative-only, like
// the orb's synthetic aria state: M1 has no live agent-graph data source.
const CONSTELLATION_NODES: { cx: number; cy: number; r: number; dim?: boolean; label: string; lx: number; ly: number }[] = [
  { cx: 260, cy: 90,  r: 2.6, dim: true, label: 'RRS BM',     lx: 266, ly: 86 },
  { cx: 280, cy: 180, r: 3.4, label: 'BUILDER',               lx: 286, ly: 176 },
  { cx: 220, cy: 240, r: 2.6, dim: true, label: 'ARCHITECT',  lx: 226, ly: 236 },
  { cx: 60,  cy: 315, r: 3.4, label: 'UNDERBOSS',             lx: 66,  ly: 311 },
  { cx: 130, cy: 328, r: 4.2, label: 'ORG CORE',              lx: 136, ly: 324 },
  { cx: 200, cy: 315, r: 3.4, label: 'SAAS BM',               lx: 206, ly: 311 },
  { cx: 270, cy: 328, r: 2.6, dim: true, label: 'ANTAGONIST', lx: 214, ly: 324 },
]

const CONSTELLATION_LINKS: [number, number, number, number][] = [
  [60, 315, 130, 328],
  [130, 328, 200, 315],
  [200, 315, 270, 328],
  [130, 328, 220, 240],
  [200, 315, 280, 180],
  [220, 240, 280, 180],
  [280, 180, 260, 90],
  [130, 328, 260, 90],
]

function Constellation() {
  return (
    <svg
      aria-hidden="true"
      preserveAspectRatio="xMidYMid meet"
      style={{ position: 'absolute', right: '4%', top: 0, bottom: 0, width: 340, zIndex: 1, opacity: 0.6, pointerEvents: 'none' }}
      viewBox="0 0 340 340"
    >
      {CONSTELLATION_LINKS.map(([x1, y1, x2, y2], i) => (
        <line key={i} stroke="rgba(232,162,77,0.22)" strokeWidth={1} x1={x1} x2={x2} y1={y1} y2={y2} />
      ))}
      {CONSTELLATION_NODES.map((n, i) => (
        <g key={n.label} style={{ animation: `cc-nodePulse 2.6s ease-in-out infinite`, animationDelay: i % 4 === 3 ? '1.2s' : i % 3 === 2 ? '.6s' : undefined }}>
          <circle
            cx={n.cx}
            cy={n.cy}
            fill={n.dim ? 'rgba(232,162,77,0.35)' : 'var(--cc-amber)'}
            r={n.r}
            style={n.dim ? undefined : { filter: 'drop-shadow(0 0 4px rgba(232,162,77,0.7))' }}
          />
        </g>
      ))}
      {CONSTELLATION_NODES.map(n => (
        <text
          fill="rgba(232,162,77,0.45)"
          fontFamily="var(--cc-font-mono)"
          fontSize="7.5px"
          key={`${n.label}-label`}
          letterSpacing="0.5px"
          x={n.lx}
          y={n.ly}
        >
          {n.label}
        </text>
      ))}
    </svg>
  )
}

// ── Orb component ────────────────────────────────────────────────────────────
function Orb({ ariaLabel }: { ariaLabel: string }) {
  const particlesRef = useRef<HTMLDivElement>(null)
  const degreeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Degree ring tick marks
    const dr = degreeRef.current

    if (dr) {
      dr.innerHTML = ''

      for (let deg = 0; deg < 360; deg += 6) {
        const m = document.createElement('div')
        m.className = 'cc-orb-mark' + (deg % 30 === 0 ? ' major' : '')
        m.style.cssText = `position:absolute;left:50%;top:0;width:${deg%30===0?'1.4':'1'}px;height:${deg%30===0?'10':'6'}px;background:rgba(232,162,77,${deg%30===0?'0.7':'0.4'});transform-origin:50% 164px;transform:rotate(${deg}deg);`
        dr.appendChild(m)
      }
    }

    // Particle field
    const pc = particlesRef.current

    if (pc) {
      pc.innerHTML = ''

      for (let i = 0; i < 22; i++) {
        const p = document.createElement('div')
        p.style.cssText = 'position:absolute;width:2px;height:2px;border-radius:50%;background:var(--cc-amber);opacity:0;animation:cc-drift var(--dur) ease-in infinite;'
        const angle = Math.random() * Math.PI * 2
        const dist = 90 + Math.random() * 70
        const cx = 150 + Math.cos(angle) * 40
        const cy = 150 + Math.sin(angle) * 40
        p.style.left = cx + 'px'
        p.style.top = cy + 'px'
        p.style.setProperty('--dx', (Math.cos(angle) * dist) + 'px')
        p.style.setProperty('--dy', (Math.sin(angle) * dist) + 'px')
        p.style.setProperty('--dur', (3.5 + Math.random() * 2.5) + 's')
        p.style.animationDelay = (Math.random() * 4.5) + 's'
        pc.appendChild(p)
      }
    }
  }, [])

  // Orb is always UNKNOWN/neutral in M1 — no spectrumCycle, no attention animation
  return (
    <div
      aria-label={ariaLabel}
      role="img"
      style={{ position: 'relative', width: 300, height: 300, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
    >
      {/* Rings */}
      {[
        { inset: -14, dur: '60s', dash: false, color: 'rgba(232,162,77,0.08)' },
        { inset: 0,   dur: '34s', dash: true,  color: 'rgba(232,162,77,0.16)' },
        { inset: 22,  dur: '22s', dash: false, color: 'rgba(232,162,77,0.3)', rev: true },
        { inset: 48,  dur: '40s', dash: true,  color: 'rgba(232,162,77,0.14)' },
      ].map((r, i) => (
        <div key={i} style={{ position: 'absolute', borderRadius: '50%', border: `1px ${r.dash ? 'dashed' : 'solid'} ${r.color}`, inset: r.inset, animation: `cc-spin ${r.dur} linear infinite${r.rev ? ' reverse' : ''}` }} />
      ))}
      {/* Degree ring */}
      <div ref={degreeRef} style={{ position: 'absolute', inset: -14, borderRadius: '50%', animation: 'cc-spin 90s linear infinite' }} />
      {/* Arcs */}
      {[
        { inset: 8,  color: 'var(--cc-amber)', dur: '6s', dir: '' },
        { inset: 34, color: 'var(--cc-amber)', dur: '9s', dir: 'reverse', side: 'right' },
        { inset: 60, color: 'rgba(255,224,168,0.9)', dur: '5s', dir: '', sides: 'bottom left' },
      ].map((a, i) => (
        <div key={i} style={{ position: 'absolute', inset: a.inset, borderRadius: '50%', border: '2px solid transparent', borderTopColor: i === 0 ? a.color : 'transparent', borderRightColor: i === 1 ? a.color : undefined, borderBottomColor: i === 2 ? a.color : undefined, borderLeftColor: i === 2 ? a.color : undefined, opacity: i === 0 ? 0.7 : i === 1 ? 0.45 : 0.4, animation: `cc-spin ${a.dir === 'reverse' ? '9s linear infinite reverse' : `${a.dir === '' && i === 0 ? '6' : '5'}s linear infinite`}`, filter: 'drop-shadow(0 0 8px var(--cc-amber-glow))' }} />
      ))}
      {/* Core */}
      <div style={{ position: 'relative', width: 104, height: 104, borderRadius: '50%', background: 'radial-gradient(circle at 38% 32%, #fffaf0 0%, #ffd08a 18%, var(--cc-amber) 42%, #8a5b23 78%, #1a1206 100%)', boxShadow: '0 0 34px 8px var(--cc-amber-glow), 0 0 90px 22px rgba(232,162,77,0.16), inset 0 0 30px rgba(0,0,0,0.4), inset 0 0 3px 1px rgba(255,255,255,0.5)', animation: 'cc-pulse 3.2s ease-in-out infinite', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: -6, borderRadius: '50%', border: '1px solid rgba(255,240,214,0.5)' }} />
        {/* Waveform */}
        <div style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%,-50%)', display: 'flex', alignItems: 'center', gap: '2.4px', zIndex: 3 }}>
          {[6,14,20,11,17,8,15].map((h, i) => (
            <div key={i} style={{ width: '2.4px', height: h, background: 'linear-gradient(180deg,#1c0f02,#3a2004)', borderRadius: 2, animation: `cc-waveBounce 1.1s ease-in-out infinite`, animationDelay: `${i * 0.12}s` }} />
          ))}
        </div>
      </div>
      {/* UNKNOWN label — always shown in M1 */}
      <div style={{ position: 'absolute', bottom: -8, left: '50%', transform: 'translateX(-50%)', fontFamily: 'var(--cc-font-mono)', fontSize: '9.5px', letterSpacing: '1.2px', textTransform: 'uppercase', color: 'var(--cc-text-faint)', whiteSpace: 'nowrap' }}>
        STATE UNKNOWN
      </div>
      {/* Particle container */}
      <div ref={particlesRef} style={{ position: 'absolute', inset: 0, borderRadius: '50%', pointerEvents: 'none' }} />
    </div>
  )
}

// ── Home screen ───────────────────────────────────────────────────────────────
function HomeScreen({ onNavigate, onOpenDecision }: { onNavigate: (s: Screen) => void; onOpenDecision: (item: AttentionItem) => void }) {
  const { data: summaryEnv } = useQuery({ queryKey: ccKeys.summary, queryFn: fetchSummary, refetchInterval: POLL_SUMMARY_MS })
  const { data: workEnv }    = useQuery({ queryKey: ccKeys.work,    queryFn: fetchWork,    refetchInterval: POLL_WORK_MS })
  const { data: orbEnv }     = useQuery({ queryKey: ccKeys.orbState, queryFn: fetchOrbState, refetchInterval: POLL_ORB_MS })

  const { data: summary } = unwrap(summaryEnv)
  const { data: work }    = unwrap(workEnv)
  const { data: orb }     = unwrap(orbEnv)

  const decisionCount = summary?.attention.decision_count ?? 0
  const actionCount   = summary?.attention.action_count ?? 0
  const orbAria = orb?.aria_label ?? 'Organizational state: unknown. Live data not connected in M1.'

  return (
    <div className="cc-main">
      {/* Sub-filter bar */}
      <div aria-label="Attention filters" className="cc-subbar" role="tablist">
        {['All attention', 'Decisions', 'Actions', 'Information', 'By agent', 'Snoozed'].map((f, i) => (
          <span aria-selected={i === 0} className={`cc-subbar-item${i === 0 ? ' active' : ''}`} key={f} role="tab">{f}</span>
        ))}
      </div>

      {/* Hero */}
      <section style={{ position: 'relative', display: 'grid', gridTemplateColumns: '320px 1fr', alignItems: 'center', gap: 34, padding: '30px 56px 26px', borderBottom: '1px solid var(--cc-border-soft)', overflow: 'hidden', flexShrink: 0, minHeight: 340, background: '#07070a' }}>
        {/* Background atmosphere */}
        <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(700px 460px at 14% 40%, rgba(232,162,77,0.13), transparent 62%), radial-gradient(620px 420px at 88% 55%, rgba(232,162,77,0.08), transparent 62%)' }} />
          <div style={{ position: 'absolute', inset: -2, backgroundImage: 'linear-gradient(rgba(232,162,77,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(232,162,77,0.05) 1px, transparent 1px)', backgroundSize: '34px 34px', WebkitMaskImage: 'radial-gradient(1100px 520px at 55% 50%, black 20%, transparent 82%)', animation: 'cc-gridDrift 26s linear infinite' }} />
        </div>
        {/* HUD corners */}
        {[['tl','top','left'],['tr','top','right'],['bl','bottom','left'],['br','bottom','right']].map(([,v,h]) => (
          <div key={`${v}${h}`} style={{ position: 'absolute', width: 26, height: 26, borderColor: 'rgba(232,162,77,0.35)', zIndex: 1, [v === 'top' ? 'top' : 'bottom']: 14, [h === 'left' ? 'left' : 'right']: 14, [`border${v === 'top' ? 'Top' : 'Bottom'}`]: '1.5px solid', [`border${h === 'left' ? 'Left' : 'Right'}`]: '1.5px solid' }} />
        ))}

        {/* Right-side agent constellation (a-hero/index.html) */}
        <Constellation />

        {/* Orb */}
        <div style={{ position: 'relative', zIndex: 2, justifySelf: 'start' }}>
          <Orb ariaLabel={orbAria} />
        </div>

        {/* Hero copy */}
        <div style={{ position: 'relative', zIndex: 2, minWidth: 0 }}>
          <div className="cc-eyebrow">Command Center · {new Date().toLocaleDateString('en-US', { weekday: 'long', day: 'numeric', month: 'long' })}</div>
          <h1 style={{ fontFamily: 'var(--cc-font-display)', fontWeight: 600, fontSize: 38, letterSpacing: '-0.4px', lineHeight: 1.08, margin: 0 }}>
            What needs your attention.
          </h1>
          <div style={{ color: 'var(--cc-text-dim)', fontSize: 14.5, marginTop: 10, maxWidth: 560, lineHeight: 1.55 }}>
            {summary
              ? `${summary.attention.decision_count} decision${summary.attention.decision_count !== 1 ? 's' : ''}, ${summary.attention.action_count} action${summary.attention.action_count !== 1 ? 's' : ''}, ${summary.attention.information_count} thing${summary.attention.information_count !== 1 ? 's' : ''} worth knowing.`
              : 'Loading organizational summary…'}
          </div>

          {/* Command bar */}
          <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', gap: 12, maxWidth: 620, border: '1px solid var(--cc-border)', background: 'rgba(22,22,27,0.7)', backdropFilter: 'blur(6px)', borderRadius: 11, padding: '13px 16px' }}>
            <span style={{ color: 'var(--cc-amber-dim)', fontSize: 15 }}>⌘</span>
            <input placeholder="Give the organization an objective, or ask a question…" readOnly style={{ all: 'unset' as any, flex: 1, color: 'var(--cc-text)', fontSize: 14, fontFamily: 'var(--cc-font-body)' }} />
            <span style={{ fontFamily: 'var(--cc-font-mono)', fontSize: '10.5px', color: 'var(--cc-text-faint)', border: '1px solid var(--cc-border)', borderRadius: 5, padding: '2px 6px' }}>⌘K</span>
          </div>

          {/* Telemetry strip (§2.6) */}
          <div style={{ display: 'flex', gap: 26, marginTop: 20 }}>
            {[
              { n: summary ? `${summary.agents.live}/${summary.agents.total_known}` : '—', l: 'Agents live' },
              { n: summary ? String(decisionCount + actionCount) : '—', l: 'Needs you', warn: (decisionCount + actionCount) > 0 },
              { n: summary ? `${summary.objectives.on_track}/${summary.objectives.total}` : '—', l: 'Objectives on track' },
              { n: summary?.system_health.state === 'degraded' ? 'Degraded' : 'Unknown', l: 'System health', faint: true },
            ].map(s => (
              <div key={s.l} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontFamily: 'var(--cc-font-display)', fontSize: 20, fontWeight: 600, color: s.warn ? 'var(--cc-amber)' : s.faint ? 'var(--cc-text-faint)' : undefined }}>{s.n}</span>
                <span style={{ fontFamily: 'var(--cc-font-mono)', fontSize: '9.5px', letterSpacing: '0.8px', textTransform: 'uppercase', color: 'var(--cc-text-faint)' }}>{s.l}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Content below hero */}
      <div style={{ padding: '26px 40px 60px', display: 'grid', gridTemplateColumns: '1fr 320px', gap: 34 }}>
        {/* Attention feed */}
        <div>
          <div className="cc-section-label">Attention</div>
          {work
            ? work.attention_items.map(item => (
                <AttentionCard item={item} key={item.id} onAction={item.classification !== 'information' ? onOpenDecision : undefined} />
              ))
            : <Placeholder label="Loading attention items…" />}
        </div>

        {/* Sidebar — objectives + agents */}
        <div>
          <div className="cc-section-label">Objectives</div>
          <div className="cc-card" style={{ marginBottom: 14 }}>
            {work?.objectives.map(obj => <ObjectiveRow key={obj.id} obj={obj} />) ?? <Placeholder />}
          </div>

          <div className="cc-section-label" style={{ marginTop: 6 }}>Agents</div>
          <div className="cc-card">
            {work
              ? [
                  { name: 'Underboss', doing: 'Awaiting your sign-off', color: 'var(--cc-green)' },
                  { name: 'RRS Business Manager', doing: 'Reviewing 4 candidates', color: 'var(--cc-green)' },
                  { name: 'SaaS Business Manager', doing: 'Blocked — needs your decision', color: 'var(--cc-amber)' },
                  { name: 'Architect', doing: 'Idle', color: 'var(--cc-text-faint)' },
                ].map(a => (
                  <div key={a.name} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 0', borderBottom: '1px solid var(--cc-border-soft)' }}>
                    <div aria-hidden="true" className="cc-avatar-lg">
                      <span aria-hidden="true" className="cc-status-dot" style={{ background: a.color }} />
                    </div>
                    <div>
                      <div style={{ fontSize: '12.5px', fontWeight: 600 }}>{a.name}</div>
                      <div style={{ fontSize: 11, color: 'var(--cc-text-faint)' }}>{a.doing}</div>
                    </div>
                  </div>
                ))
              : <Placeholder />}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Goals screen ──────────────────────────────────────────────────────────────
function GoalsScreen() {
  const { data: workEnv } = useQuery({ queryKey: ccKeys.work, queryFn: fetchWork, refetchInterval: POLL_WORK_MS })
  const { data: work }    = unwrap(workEnv)

  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Command Center · Goals & Outcomes</div>
        <h1 className="cc-h1">Your organization's progress, at a glance.</h1>
        <div className="cc-subhead">Objective → Project → Workflow → Task. Progress is outcome-first; task detail is a drill-down.</div>
      </div>
      <div className="cc-page-content">
        <GateNote>All progress data is synthetic (M1). Real Hermes integration is not connected.</GateNote>
        <div className="cc-section-label">Active objectives</div>
        <div className="cc-card">
          {work?.objectives.map(obj => <ObjectiveRow key={obj.id} obj={obj} />) ?? <Placeholder />}
        </div>
      </div>
    </div>
  )
}

// ── Agents Directory screen ───────────────────────────────────────────────────
function AgentsDirectoryScreen({
  onSelectAgent,
  onOpenDirectRoster,
  onOpenUnderboss,
}: {
  onSelectAgent: (id: string) => void
  onOpenDirectRoster: () => void
  onOpenUnderboss: () => void
}) {
  const { data: agentsEnv } = useQuery({ queryKey: ccKeys.agents, queryFn: fetchAgents, refetchInterval: POLL_AGENTS_MS })
  const { data: agentsData } = unwrap(agentsEnv)
  const agents = agentsData?.agents ?? []
  const live = agents.filter(a => a.implementation_status === 'live')
  const proposed = agents.filter(a => a.implementation_status === 'proposed')

  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Command Center · Agents</div>
        <h1 className="cc-h1">Every agent, real and proposed — clearly labeled.</h1>
        <div className="cc-subhead">Live/Owned agents are running. Proposed/Unbuilt agents are documented target architecture only — no capability implied.</div>
      </div>
      <div className="cc-page-content">
        <GateNote>Agent activity data is synthetic (M1). Real gateway not connected.</GateNote>

        <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
          <button className="cc-btn primary" onClick={onOpenDirectRoster} type="button">
            Talk to an agent directly →
          </button>
          <button className="cc-btn ghost" onClick={onOpenUnderboss} type="button">
            Talk to Underboss →
          </button>
        </div>

        <div className="cc-section-label">Live agents ({live.length})</div>
        <div className="cc-grid-2" style={{ gap: 14 }}>
          {live.map(a => (
            <AgentCard focus={a.current_focus} implementationStatus={a.implementation_status} key={a.id} liveStatus={a.live_status} name={a.display_label} onClick={() => onSelectAgent(a.id)} role={a.role_label} />
          ))}
        </div>
        <div className="cc-section-label" style={{ marginTop: 28 }}>Proposed / unbuilt ({proposed.length})</div>
        <div className="cc-grid-2" style={{ gap: 14 }}>
          {proposed.map(a => (
            <AgentCard focus={a.current_focus} implementationStatus={a.implementation_status} key={a.id} liveStatus={a.live_status} name={a.display_label} role={a.role_label} />
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Agent View screen (§5 row 2) ─────────────────────────────────────────────
function AgentViewScreen({ agentId, onBack }: { agentId: string; onBack: () => void }) {
  const { data: agentsEnv } = useQuery({ queryKey: ccKeys.agents, queryFn: fetchAgents })
  const { data: workEnv }   = useQuery({ queryKey: ccKeys.work,   queryFn: fetchWork })
  const agents = agentsEnv?.data.agents ?? []
  const agent = agents.find(a => a.id === agentId)
  const work  = workEnv?.data

  const agentDecisions = work?.attention_items.filter(
    i => i.raised_by.agent_id === agentId && (i.classification === 'decision' || i.classification === 'action')
  ) ?? []

  const isNeverContacted = !agent?.last_observed && agent?.implementation_status === 'live'
  const directChatRow = useAgentChatEligibility(agent?.id ?? '')

  return (
    <div className="cc-main">
      {/* Agent header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 18, padding: '28px 40px 0', flexShrink: 0 }}>
        <button className="cc-btn ghost" onClick={onBack} style={{ padding: '5px 10px' }} type="button">← Agents</button>
      </div>
      <div style={{ flex: 1, overflow: 'hidden', display: 'grid', gridTemplateColumns: '1fr 340px' }}>
        {/* Main column */}
        <div style={{ padding: '28px 40px', borderRight: '1px solid var(--cc-border-soft)', overflowY: 'auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 18, marginBottom: 22 }}>
            <div style={{ width: 56, height: 56, borderRadius: '50%', background: 'linear-gradient(145deg,#3a3a42,#1c1c21)', flexShrink: 0, border: '2px solid var(--cc-amber-glow)', position: 'relative' }}>
              {agent?.live_status && (
                <span style={{ position: 'absolute', bottom: 0, right: 0, width: 14, height: 14, borderRadius: '50%', background: agent.live_status === 'running' ? 'var(--cc-green)' : agent.live_status === 'blocked' ? 'var(--cc-amber)' : 'var(--cc-text-faint)', border: '3px solid var(--cc-bg)' }} />
              )}
            </div>
            <div>
              <h1 style={{ fontFamily: 'var(--cc-font-display)', fontWeight: 600, fontSize: 24, margin: 0 }}>{agent?.display_label ?? 'Agent'}</h1>
              <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 11, color: 'var(--cc-text-faint)', textTransform: 'uppercase', letterSpacing: '0.6px', marginTop: 3 }}>{agent?.role_label}</div>
            </div>
            {agent?.implementation_status === 'live' && (
              <span style={{ marginLeft: 'auto', fontFamily: 'var(--cc-font-mono)', fontSize: '9.5px', letterSpacing: '0.6px', textTransform: 'uppercase', color: 'var(--cc-green)', background: 'rgba(111,191,140,0.12)', padding: '4px 10px', borderRadius: 20 }}>
                Live · Owned
              </span>
            )}
          </div>

          {/* Since Your Last Interaction card — §5 row 2, must render every visit */}
          {isNeverContacted
            ? (
              <div style={{ background: 'linear-gradient(135deg, var(--cc-amber-glow), transparent 60%), var(--cc-bg-card)', border: '1px solid var(--cc-border)', borderRadius: 12, padding: '18px 20px', marginBottom: 26 }}>
                <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 10, letterSpacing: '1.2px', textTransform: 'uppercase', color: 'var(--cc-amber-dim)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 12, height: 1, background: 'var(--cc-amber-dim)', display: 'inline-block' }} />
                  No prior interaction on record
                </div>
                <p style={{ fontSize: '13.5px', color: 'var(--cc-text-dim)', lineHeight: 1.6 }}>
                  You have not previously interacted directly with <strong style={{ color: 'var(--cc-text)' }}>{agent?.display_label}</strong>. Their current focus and any pending items appear in the feed below.
                </p>
              </div>
            )
            : (
              <div style={{ background: 'linear-gradient(135deg, var(--cc-amber-glow), transparent 60%), var(--cc-bg-card)', border: '1px solid var(--cc-border)', borderRadius: 12, padding: '18px 20px', marginBottom: 26 }}>
                <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 10, letterSpacing: '1.2px', textTransform: 'uppercase', color: 'var(--cc-amber-dim)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 12, height: 1, background: 'var(--cc-amber-dim)', display: 'inline-block' }} />
                  Since your last interaction · 3 days ago — <span style={{ color: 'var(--cc-text-faint)' }}>SIMULATED</span>
                </div>
                <p style={{ fontSize: '13.5px', color: 'var(--cc-text-dim)', lineHeight: 1.6 }}>
                  Since you last spoke, the <strong style={{ color: 'var(--cc-text)' }}>{agent?.current_focus}</strong>. Any decisions or actions raised appear in your attention feed.
                </p>
              </div>
            )
          }

          {/* Attention items for this agent */}
          {agentDecisions.length > 0
            ? agentDecisions.map(item => <AttentionCard item={item} key={item.id} />)
            : <div style={{ color: 'var(--cc-text-faint)', fontSize: 13, fontFamily: 'var(--cc-font-mono)' }}>No pending decisions or actions from this agent.</div>}

          {/* Direct agent chat handoff — Two Owner Communication Modes release.
              Only renders when the CURRENT authorized runtime roster confirms
              this agent id resolves to an available profile; never assumed
              from the M1 synthetic agent record alone. */}
          {directChatRow && (
            <div style={{ marginTop: 20 }}>
              <div className="cc-section-label" style={{ marginBottom: 10 }}>Talk to {agent?.display_label ?? 'this agent'} directly</div>
              <DirectAgentHandoffCard row={directChatRow} />
            </div>
          )}

          {/* Composer — present, disabled in M1 */}
          <div style={{ marginTop: 20, display: 'flex', gap: 10, alignItems: 'center', border: '1px solid var(--cc-border)', background: 'var(--cc-bg-card)', borderRadius: 11, padding: '12px 14px', opacity: 0.45 }}>
            <input disabled placeholder="Message this agent… (disabled in M1)" style={{ all: 'unset' as any, flex: 1, fontSize: '13.5px', color: 'var(--cc-text)', fontFamily: 'var(--cc-font-body)' }} />
            <span style={{ fontFamily: 'var(--cc-font-mono)', fontSize: '10.5px', color: 'var(--cc-text-faint)', border: '1px solid var(--cc-border)', borderRadius: 5, padding: '3px 8px' }}>Enter ↵</span>
          </div>
          <GateNote>Direct agent messaging is disabled in M1. Composer shown for visual completeness.</GateNote>
        </div>

        {/* Sidebar */}
        <div style={{ padding: '28px 24px', overflowY: 'auto' }}>
          <div className="cc-section-label">Current focus</div>
          <div className="cc-card" style={{ marginBottom: 18 }}>
            <div className="cc-stat-row"><span className="k">Status</span><span className="v" style={{ color: agent?.live_status === 'blocked' ? 'var(--cc-red)' : 'var(--cc-text)' }}>{agent?.live_status ?? '—'}</span></div>
            <div className="cc-stat-row"><span className="k">Last observed</span><span className="v">{agent?.last_observed ? new Date(agent.last_observed).toLocaleTimeString() : 'Unknown'}</span></div>
          </div>
          <GateNote>Scope and authority data require real Hermes gateway integration (deferred).</GateNote>
        </div>
      </div>
    </div>
  )
}

// ── Orchestration screen ──────────────────────────────────────────────────────
function OrchestrationScreen() {
  const stages = ['Scoping', 'Investigation', 'Escalation', 'Resolution']
  const activeIdx = 2

  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Command Center · Orchestration</div>
        <h1 className="cc-h1">Active work across the enterprise.</h1>
        <div className="cc-subhead">Stage-by-stage progress — not a technical dependency graph. See Org Map for the authority structure.</div>
      </div>
      <div className="cc-page-content">
        <GateNote>Orchestration data is synthetic (M1). Real Kanban/workflow integration is not connected.</GateNote>
        <div className="cc-section-label">Velora launch — active workflow</div>
        <div className="cc-card" style={{ marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            {stages.map((s, i) => (
              <Fragment key={s}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: i < activeIdx ? 'var(--cc-green)' : i === activeIdx ? 'var(--cc-amber)' : 'var(--cc-border)', flexShrink: 0 }} />
                  <span style={{ fontSize: 12, color: i === activeIdx ? 'var(--cc-text)' : i < activeIdx ? 'var(--cc-green)' : 'var(--cc-text-faint)', fontWeight: i === activeIdx ? 600 : undefined }}>{s}</span>
                </div>
                {i < stages.length - 1 && <span style={{ flex: 1, height: 1, background: i < activeIdx ? 'var(--cc-green)' : 'var(--cc-border)' }} />}
              </Fragment>
            ))}
          </div>
          <div className="cc-stat-row"><span className="k">Current stage</span><span className="v" style={{ color: 'var(--cc-amber)' }}>Escalation</span></div>
          <div className="cc-stat-row"><span className="k">Raised by</span><span className="v">SaaS Business Manager</span></div>
          <div className="cc-stat-row"><span className="k">Awaiting</span><span className="v">Owner scope decision</span></div>
        </div>
        <div className="cc-section-label">Recently completed</div>
        <div className="cc-grid-2">
          <div className="cc-card">
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Veritus milestone review</div>
            <div style={{ fontSize: 11, color: 'var(--cc-text-faint)' }}>Completed Aug 24 · SaaS BM</div>
            <div style={{ marginTop: 8 }}><StatusChip label="Done" variant="canonical" /></div>
          </div>
          <div className="cc-card">
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>RRS candidate pipeline refresh</div>
            <div style={{ fontSize: 11, color: 'var(--cc-text-faint)' }}>Completed Aug 22 · RRS BM</div>
            <div style={{ marginTop: 8 }}><StatusChip label="Done" variant="canonical" /></div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Knowledge screen ──────────────────────────────────────────────────────────
function KnowledgeScreen() {
  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Command Center · Knowledge</div>
        <h1 className="cc-h1">Search what you're authorized to see.</h1>
        <div className="cc-subhead">Owner → Command Center → Knowledge Manager → Vault. This gate is never bypassed, including by this interface itself.</div>
      </div>
      <div className="cc-page-content">
        {/* Gate note — required per spec §5 row 7, must always be visible */}
        <GateNote>
          Vault access routes through the Knowledge Manager (proposed target role, not yet implemented). Results below are illustrative of the intended experience, not live. No direct EKV or filesystem path is accessed by this screen.
        </GateNote>

        {/* Search field — non-functional in M1 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, border: '1px solid var(--cc-border)', background: 'var(--cc-bg-card)', borderRadius: 11, padding: '13px 16px', marginBottom: 20, opacity: 0.55 }}>
          <span style={{ color: 'var(--cc-amber-dim)' }}>⌕</span>
          <input disabled placeholder="Search documents, decisions, agent knowledge, project history… (disabled in M1)" style={{ all: 'unset' as any, flex: 1, color: 'var(--cc-text)', fontSize: 14, fontFamily: 'var(--cc-font-body)' }} />
        </div>

        <div className="cc-section-label">Illustrative — Command Center project</div>
        <div className="cc-card" style={{ padding: 0 }}>
          {[
            { title: 'Enterprise Architecture V1.0', chip: 'canonical' as const, date: '2026-08-23' },
            { title: 'Command Center — Owner Charter v0.1', chip: 'proposed' as const, date: '2026-08-27' },
            { title: 'Track G — Authority Model', chip: 'canonical' as const, date: '2026-08-26' },
            { title: 'Enterprise Agent Architecture Owner Review', chip: 'superseded' as const, date: '2026-08-22' },
            { title: 'Business Canonical Status', chip: 'unresolved' as const, date: '2026-08-22' },
          ].map(row => (
            <div key={row.title} style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '13px 16px', borderBottom: '1px solid var(--cc-border-soft)' }}>
              <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>{row.title}</span>
              <StatusChip variant={row.chip} />
              <span style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 10, color: 'var(--cc-text-faint)' }}>{row.date}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Org Map screen ────────────────────────────────────────────────────────────
function OrgMapScreen() {
  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Command Center · Org Map</div>
        <h1 className="cc-h1">Who has what authority.</h1>
        <div className="cc-subhead">SVG hierarchy of the enterprise authority structure. Nodes show implementation status. This is the authority model — not a live dependency graph.</div>
      </div>
      <div className="cc-page-content">
        <GateNote>Org Map is illustrative in M1. Real authority data requires Syntex integration.</GateNote>
        <svg aria-label="Enterprise authority hierarchy diagram" style={{ width: '100%', maxWidth: 700, display: 'block', margin: '0 auto' }} viewBox="0 0 600 340">
          {/* Owner → Underboss → Specialists */}
          <line stroke="rgba(232,162,77,0.3)" strokeWidth="1" x1="300" x2="300" y1="40" y2="90" />
          <line stroke="rgba(232,162,77,0.2)" strokeWidth="1" x1="300" x2="120" y1="130" y2="200" />
          <line stroke="rgba(232,162,77,0.2)" strokeWidth="1" x1="300" x2="300" y1="130" y2="200" />
          <line stroke="rgba(232,162,77,0.2)" strokeWidth="1" x1="300" x2="480" y1="130" y2="200" />
          {/* Nodes */}
          {[
            { x: 300, y: 20, label: 'Owner', chip: 'live' as const },
            { x: 300, y: 110, label: 'Underboss', chip: 'live' as const },
            { x: 80,  y: 220, label: 'RRS BM', chip: 'live' as const },
            { x: 300, y: 220, label: 'SaaS BM', chip: 'live' as const },
            { x: 480, y: 220, label: 'Architect', chip: 'live' as const },
          ].map(n => (
            <g key={n.label}>
              <circle cx={n.x} cy={n.y} fill="var(--cc-bg-card)" r={16} stroke="rgba(232,162,77,0.5)" strokeWidth="1.5" />
              <text style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 9, fill: 'rgba(232,162,77,0.7)', letterSpacing: '0.5px' }} textAnchor="middle" x={n.x} y={n.y + 30}>{n.label}</text>
            </g>
          ))}
          {/* Legend */}
          <g>
            <circle cx={40} cy={310} fill="rgba(111,191,140,0.7)" r={5} />
            <text style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 8, fill: 'var(--cc-text-faint)' }} x={52} y={314}>Live</text>
            <circle cx={100} cy={310} fill="rgba(232,162,77,0.5)" r={5} />
            <text style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 8, fill: 'var(--cc-text-faint)' }} x={112} y={314}>Proposed</text>
            <circle cx={175} cy={310} fill="var(--cc-text-faint)" r={5} />
            <text style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 8, fill: 'var(--cc-text-faint)' }} x={187} y={314}>Unavailable</text>
          </g>
        </svg>
      </div>
    </div>
  )
}

// ── History screen ────────────────────────────────────────────────────────────
function HistoryScreen() {
  const events = [
    { id: 'h1', type: 'decision', label: 'Decision · Owner approved Velora milestone', when: '2026-08-24 · 3:14 PM', color: 'var(--cc-red)' },
    { id: 'h2', type: 'approval', label: 'Approval · Engineering priority set', when: '2026-08-19 · 10:02 AM', color: 'var(--cc-amber)' },
    { id: 'h3', type: 'milestone', label: 'Milestone · RRS Q3 target entered on-track range', when: '2026-08-17 · 4:30 PM', color: 'var(--cc-green)' },
    { id: 'h4', type: 'neutral', label: 'Nightly synthesis · 14 changes, no escalation', when: '2026-08-16 · 12:00 AM', color: 'var(--cc-text-faint)' },
  ]

  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Command Center · History & Activity</div>
        <h1 className="cc-h1">A record of what happened and why.</h1>
        <div className="cc-subhead">Filtered subset — decisions, approvals, and meaningful milestones. Full audit record available on request.</div>
      </div>
      <div className="cc-page-content">
        <GateNote>History is illustrative in M1. Full audit feed requires Hermes integration.</GateNote>
        <div className="cc-section-label">Recent</div>
        <div style={{ position: 'relative' }}>
          <div style={{ position: 'absolute', left: 7, top: 0, bottom: 0, width: 1, background: 'var(--cc-border-soft)' }} />
          {events.map(e => (
            <div key={e.id} style={{ display: 'flex', gap: 16, marginBottom: 18, paddingLeft: 24, position: 'relative' }}>
              <span style={{ position: 'absolute', left: 0, top: 4, width: 14, height: 14, borderRadius: '50%', background: e.color, flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{e.label}</div>
                <div style={{ fontFamily: 'var(--cc-font-mono)', fontSize: 10, color: 'var(--cc-text-faint)', marginTop: 3 }}>{e.when}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── System Health screen ──────────────────────────────────────────────────────
function HealthScreen() {
  const { data: summaryEnv } = useQuery({ queryKey: ccKeys.summary, queryFn: fetchSummary, refetchInterval: POLL_SUMMARY_MS })
  const { data: summary }    = unwrap(summaryEnv)

  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Command Center · System Health</div>
        <h1 className="cc-h1">Owner-friendly by default.</h1>
        <div className="cc-subhead">Conceptually separate from organizational work. Consequential technical problems surface in Attention too — this is the full picture.</div>
      </div>
      <div className="cc-page-content">
        <GateNote>All health data is synthetic (M1). Real system metrics require Hermes gateway integration.</GateNote>
        <div className="cc-grid-4" style={{ marginBottom: 22 }}>
          <div className="cc-health-tile">
            <div className="hk">Agents</div>
            <div className={`hv ${summary ? 'ok' : 'unknown'}`}>{summary ? `${summary.agents.live} / ${summary.agents.total_known} live` : 'Unknown'}</div>
            <div className="hsub">{summary?.agents.blocked ?? 0} proposed, unbuilt</div>
          </div>
          <div className="cc-health-tile">
            <div className="hk">Model providers</div>
            <div className="hv ok">3 / 3 up</div>
            <div className="hsub">All responding normally</div>
          </div>
          <div className="cc-health-tile">
            <div className="hk">Vault sync</div>
            <div className="hv warn">Unknown</div>
            <div className="hsub">Knowledge Manager not connected</div>
          </div>
          <div className="cc-health-tile">
            <div className="hk">Integrations</div>
            <div className="hv err">1 down</div>
            <div className="hsub">Kanban proof — non-production</div>
          </div>
        </div>

        <div className="cc-section-label">Affecting organizational work</div>
        <div className="cc-card" style={{ marginBottom: 14 }}>
          <div className="cc-stat-row"><span className="k">Kanban snapshot route degraded</span><span className="v" style={{ color: 'var(--cc-amber)' }}>Non-production only</span></div>
          <div className="cc-stat-row"><span className="k">No automatic remediation authorized</span><span className="v" style={{ color: 'var(--cc-text-faint)' }}>—</span></div>
        </div>

        <div className="cc-section-label">Technical detail</div>
        <div className="cc-card">
          <div className="cc-stat-row"><span className="k">Active sessions</span><span className="v">Synthetic M1 — not real</span></div>
          <div className="cc-stat-row"><span className="k">Latency, p50</span><span className="v">—</span></div>
          <div className="cc-stat-row"><span className="k">Last deploy</span><span className="v">—</span></div>
        </div>
      </div>
    </div>
  )
}

// ── Briefings screen ──────────────────────────────────────────────────────────
function BriefingsScreen() {
  const [depth, setDepth] = useState<'quick' | 'standard' | 'full'>('standard')

  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Command Center · Briefings</div>
        <h1 className="cc-h1">The situation, at your preferred depth.</h1>
        <div className="cc-subhead">Three depths: 30-second, Standard (default), Full. Scheduled briefings require explicit Owner configuration — never auto-enabled.</div>
      </div>
      <div className="cc-page-content">
        <GateNote>Briefing content is synthetic (M1). Real intelligence synthesis requires Hermes integration.</GateNote>
        <div style={{ display: 'flex', gap: 8, marginBottom: 22 }}>
          {(['quick','standard','full'] as const).map(d => (
            <button className={`cc-btn ${depth === d ? 'primary' : 'ghost'}`} key={d} onClick={() => setDepth(d)} type="button">
              {d === 'quick' ? '30-second' : d === 'standard' ? 'Standard' : 'Full'}
            </button>
          ))}
        </div>

        <div className="cc-section-label">Today's briefing — {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}</div>
        <div className="cc-card" style={{ marginBottom: 14 }}>
          {depth === 'quick' && (
            <p style={{ fontSize: 14, lineHeight: 1.6 }}>
              <strong>2 decisions</strong> need your judgment today. Velora launch at risk — scope vs. date call pending. RRS outreach objective proposed. Everything else running quietly.
            </p>
          )}
          {depth === 'standard' && (
            <>
              <p style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 12 }}>
                <strong>Velora (SaaS):</strong> 9-day slip. EMR sync vs. launch date decision pending your input. Evidence high, confidence medium due to agent disagreement.
              </p>
              <p style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 12 }}>
                <strong>RRS Q3:</strong> On track at 81%. New clinician outreach objective proposed by RRS BM.
              </p>
              <p style={{ fontSize: 14, lineHeight: 1.6 }}>
                <strong>Restoration:</strong> Membership growth on track. Content cascade proceeding.
              </p>
            </>
          )}
          {depth === 'full' && (
            <p style={{ fontSize: 13, color: 'var(--cc-text-dim)', lineHeight: 1.7 }}>
              Full briefing with all activity, decisions, and context would appear here. In M1 this is a synthetic placeholder. Real briefing generation requires Underboss synthesis capability and Hermes gateway integration.
            </p>
          )}
        </div>

        <div className="cc-section-label">Scheduled briefings</div>
        <div className="cc-card">
          <div className="cc-stat-row"><span className="k">Scheduled briefings</span><span className="v" style={{ color: 'var(--cc-text-faint)' }}>None configured</span></div>
        </div>
        <GateNote>Scheduled briefings require explicit Owner configuration per Charter §20 and are never auto-enabled.</GateNote>
      </div>
    </div>
  )
}

// ── Security screen ───────────────────────────────────────────────────────────
function SecurityScreen() {
  const { data: secEnv } = useQuery({ queryKey: ccKeys.security, queryFn: fetchSecurity, refetchInterval: POLL_SECURITY_MS })
  const { data: sec }    = unwrap(secEnv)

  return (
    <div className="cc-main">
      <div className="cc-page-header">
        <div className="cc-eyebrow">Command Center · Security</div>
        <h1 className="cc-h1">Behind the scenes, until it matters.</h1>
        <div className="cc-subhead">What happened → why it matters → what's needed. No unnecessary technical noise.</div>
      </div>
      <div className="cc-page-content">
        <GateNote>Security events are synthetic (M1). No remediation, configuration, or credential authority is granted by this screen.</GateNote>
        {sec?.events.map(ev => (
          <div className="cc-sec-event" key={ev.id}>
            <div className="what">What happened</div>
            <div className="why">{ev.what}</div>
            <div className="what" style={{ marginTop: 10 }}>Why it matters</div>
            <div className="why">{ev.why_it_matters}</div>
            <div className="need">What's needed: {ev.what_is_needed}</div>
          </div>
        )) ?? <Placeholder />}
        <div className="cc-section-label">Nothing else to report</div>
        <div className="cc-card">
          <div className="cc-stat-row"><span className="k">Access boundary violations, 7 days</span><span className="v" style={{ color: 'var(--cc-green)' }}>{sec?.summary.access_boundary_violations ?? '—'}</span></div>
          <div className="cc-stat-row"><span className="k">Credential scope changes, 7 days</span><span className="v">{sec?.summary.credential_scope_changes ?? '—'} (blocked, above)</span></div>
          <div className="cc-stat-row"><span className="k">External integration connections</span><span className="v" style={{ color: 'var(--cc-text-faint)' }}>None active</span></div>
        </div>
      </div>
    </div>
  )
}

// ── Root page component ───────────────────────────────────────────────────────
export function OwnerCommandCenterPage() {
  const [screen, setScreen] = useState<Screen>('home')
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)
  const [approvalItem, setApprovalItem] = useState<AttentionItem | null>(null)
  // Two Owner Communication Modes: local sub-views within 'agents', not new
  // top-nav Screen values — the approved top nav is frozen. 'direct-roster'
  // is the eligible Agent Directory (Mode 2); 'underboss' is the Mode 1
  // global orchestration entry.
  const [agentSubview, setAgentSubview] = useState<'directory' | 'direct-roster' | 'underboss'>('directory')
  const [selectedDirectRow, setSelectedDirectRow] = useState<DirectAgentRow | null>(null)

  const { data: workEnv } = useQuery({ queryKey: ccKeys.work, queryFn: fetchWork, refetchInterval: POLL_WORK_MS })
  const decisionCount = workEnv?.data.attention_items.filter(i => i.classification === 'decision').length ?? 0
  const actionCount   = workEnv?.data.attention_items.filter(i => i.classification === 'action').length ?? 0

  const resetAgentsSubviews = (s: Screen) => {
    setScreen(s)
    setSelectedAgentId(null)
    setAgentSubview('directory')
    setSelectedDirectRow(null)
  }

  // Decision approval modal overlay
  if (approvalItem) {
    return (
      <div className="cc-root cc-shell">
        <TopBar actionCount={actionCount} active={screen} decisionCount={decisionCount} onNavigate={setScreen} />
        <DecisionApprovalScreen item={approvalItem} onClose={() => setApprovalItem(null)} />
      </div>
    )
  }

  // Agent view
  if (screen === 'agents' && selectedAgentId) {
    return (
      <div className="cc-root cc-shell">
        <TopBar actionCount={actionCount} active={screen} decisionCount={decisionCount} onNavigate={resetAgentsSubviews} />
        <AgentViewScreen agentId={selectedAgentId} onBack={() => setSelectedAgentId(null)} />
      </div>
    )
  }

  // Mode 2 — Direct Agent Roster
  if (screen === 'agents' && agentSubview === 'direct-roster') {
    return (
      <div className="cc-root cc-shell">
        <TopBar actionCount={actionCount} active={screen} decisionCount={decisionCount} onNavigate={resetAgentsSubviews} />
        <div style={{ padding: '20px 40px 0' }}>
          <button className="cc-btn ghost" onClick={() => { setAgentSubview('directory'); setSelectedDirectRow(null) }} style={{ padding: '5px 10px' }} type="button">← Agents</button>
        </div>
        <DirectAgentRosterScreen onSelectAgent={row => setSelectedDirectRow(row)} />
        {selectedDirectRow && (
          <div className="cc-page-content" style={{ paddingTop: 0 }}>
            <div className="cc-card" style={{ maxWidth: 480 }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 10 }}>{selectedDirectRow.handle}</div>
              <DirectAgentHandoffCard row={selectedDirectRow} />
            </div>
          </div>
        )}
      </div>
    )
  }

  // Mode 1 — Underboss global orchestration entry
  if (screen === 'agents' && agentSubview === 'underboss') {
    return (
      <div className="cc-root cc-shell">
        <TopBar actionCount={actionCount} active={screen} decisionCount={decisionCount} onNavigate={resetAgentsSubviews} />
        <div style={{ padding: '20px 40px 0' }}>
          <button className="cc-btn ghost" onClick={() => setAgentSubview('directory')} style={{ padding: '5px 10px' }} type="button">← Agents</button>
        </div>
        <UnderbossEntryScreen />
      </div>
    )
  }

  return (
    <div className="cc-root cc-shell">
      <TopBar actionCount={actionCount} active={screen} decisionCount={decisionCount} onNavigate={resetAgentsSubviews} />
      {screen === 'home'          && <HomeScreen onNavigate={setScreen} onOpenDecision={setApprovalItem} />}
      {screen === 'goals'         && <GoalsScreen />}
      {screen === 'agents'        && (
        <AgentsDirectoryScreen
          onOpenDirectRoster={() => setAgentSubview('direct-roster')}
          onOpenUnderboss={() => setAgentSubview('underboss')}
          onSelectAgent={id => setSelectedAgentId(id)}
        />
      )}
      {screen === 'orchestration' && <OrchestrationScreen />}
      {screen === 'knowledge'     && <KnowledgeScreen />}
      {screen === 'orgmap'        && <OrgMapScreen />}
      {screen === 'history'       && <HistoryScreen />}
      {screen === 'health'        && <HealthScreen />}
      {screen === 'briefings'     && <BriefingsScreen />}
      {screen === 'security'      && <SecurityScreen />}
    </div>
  )
}
