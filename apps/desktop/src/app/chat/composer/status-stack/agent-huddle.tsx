import './agent-huddle.css'

import { type CSSProperties, useMemo } from 'react'

import { useMediaQuery } from '@/hooks/use-media-query'
import { useSessionSlice } from '@/lib/use-session-slice'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'

/**
 * The "agent huddle" above the composer: a small, wordless scene of Czesiek,
 * Hermes and their helpers instead of the raw text of a worker's report.
 *
 * The report is NOT gone — it stays in `$subagentsBySession` and is voiced to
 * the Live agent by the announcement queue in `use-realtime-conversation`. It
 * is only never painted as text here: what the user watches is that the agents
 * are talking to each other.
 */
export type HuddleState = 'done' | 'talking' | 'thinking' | 'waiting'

const MOTION_QUERY = '(prefers-reduced-motion: reduce)'
const MAX_HELPERS = 2
const LABEL_MAX = 16

/**
 * Read the scene straight off the work store — the same source of truth the
 * status rows use, never a parallel clock:
 *   waiting  — a worker is on the roster but has not said anything yet
 *   thinking — the last thing a live worker said was a thought
 *   talking  — live work with actual activity behind it
 *   done     — every worker on the roster has settled
 */
export function huddleState(items: readonly SubagentProgress[]): HuddleState | null {
  if (!items.length) {
    return null
  }

  const active = items.filter(item => item.status === 'running' || item.status === 'queued')

  if (!active.length) {
    return 'done'
  }

  if (active.some(item => item.stream.at(-1)?.kind === 'thinking')) {
    return 'thinking'
  }

  const running = active.filter(item => item.status === 'running')

  return running.length > 0 && running.some(item => item.stream.length > 0) ? 'talking' : 'waiting'
}

const CAPTION: Record<HuddleState, string> = {
  done: 'Narada zakończona',
  talking: 'Czesiek rozmawia z Hermesem',
  thinking: 'Hermes analizuje zadanie',
  waiting: 'Czesiek i Hermes czekają na zlecenie'
}

/** A helper's own goal is a label, never the report: cut it down to a chip. */
const shortLabel = (goal: string) => {
  const line = goal.replace(/\s+/g, ' ').trim()

  return line.length > LABEL_MAX ? `${line.slice(0, LABEL_MAX - 1)}…` : line || 'Współpracownik'
}

interface HuddleFigure {
  key: string
  label: string
  role: 'helper' | 'main' | 'peer'
}

function figuresFor(items: readonly SubagentProgress[]): HuddleFigure[] {
  // Helpers are on stage only WHILE they work: a settled goal is not repeated
  // here (the status rows own that), so the finished scene is just Czesiek and
  // Hermes standing together.
  const helpers = items
    .filter(item => item.status === 'running' || item.status === 'queued')
    .slice(0, MAX_HELPERS)

  return [
    { key: 'main', label: 'Czesiek', role: 'main' },
    { key: 'peer', label: 'Hermes', role: 'peer' },
    ...helpers.map(item => ({ key: item.id, label: shortLabel(item.goal), role: 'helper' as const }))
  ]
}

export function AgentHuddle({ sessionId }: { sessionId: string }) {
  const items = useSessionSlice($subagentsBySession, sessionId)
  const reduced = useMediaQuery(MOTION_QUERY)
  const state = useMemo(() => huddleState(items), [items])
  const figures = useMemo(() => figuresFor(items), [items])

  if (!state) {
    return null
  }

  const stage: CSSProperties = { '--huddle-spread': `${(figures.length - 1) * 0.65}rem` } as CSSProperties

  return (
    <div
      className="agent-huddle font-normal"
      data-motion={reduced ? 'reduced' : 'full'}
      data-slot="agent-huddle"
      data-state={state}
      data-testid="agent-huddle"
    >
      <div aria-hidden="true" className="agent-huddle__stage" style={stage}>
        <span className="agent-huddle__link" />
        <span className="agent-huddle__packet" />
        <span className="agent-huddle__packet agent-huddle__packet--back" />
        <span className="agent-huddle__ring" />
        {figures.map((figure, index) => (
          <span
            className={`agent-huddle__figure agent-huddle__figure--${figure.role}`}
            key={figure.key}
            style={{ '--huddle-index': index } as CSSProperties}
          >
            <span className="agent-huddle__bubble">
              <span />
              <span />
              <span />
            </span>
            <span className="agent-huddle__wave">
              <span />
              <span />
              <span />
            </span>
            <span className="agent-huddle__body">
              <span className="agent-huddle__eyes">
                <span className="agent-huddle__eye" />
                <span className="agent-huddle__eye" />
              </span>
            </span>
            <span className="agent-huddle__shadow" />
            <span className="agent-huddle__name">{figure.label}</span>
          </span>
        ))}
      </div>
      <p className="agent-huddle__caption">{CAPTION[state]}</p>
    </div>
  )
}

/**
 * The mini strip the extended-transcript panel shows in place of the child's
 * raw output: same idea, one character wide. `active` mirrors "the child has
 * produced output", never the output itself.
 */
export function AgentTypingIndicator({ active }: { active: boolean }) {
  const reduced = useMediaQuery(MOTION_QUERY)

  return (
    <span
      aria-hidden="true"
      className="agent-huddle-typing"
      data-active={active}
      data-motion={reduced ? 'reduced' : 'full'}
      data-testid="agent-typing"
    >
      <span className="agent-huddle-typing__bar" />
      <span className="agent-huddle-typing__bar" />
      <span className="agent-huddle-typing__bar" />
    </span>
  )
}
