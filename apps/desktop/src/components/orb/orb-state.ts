// Canonical assistant/thread state enum + the state→visual mapping for the orb.
//
// The canonical state model in the desktop app is assistant-ui's
// `MessageStatus` type enum on every assistant message —
//   { type: 'running' }
//   | { type: 'requires-action'; reason: 'tool-calls' | 'interrupt' }
//   | { type: 'complete'; reason: 'stop' | 'unknown' }
//   | { type: 'incomplete'; reason: 'cancelled' | 'tool-calls' | 'length'
//       | 'content-filter' | 'other' | 'error'; error?: ... }
// — plus `MessagePartStatus` on reasoning/tool-call parts, layered with the
// renderer's session-scoped signals (awaiting input, provider wait,
// compaction) from `@/store/*`.
//
// `OrbState` is that model projected onto what the orb can show. `OrbView`
// takes an `OrbState` and renders `orbParamsForState(state, base)` — the
// user's orb (built-in default or BYO configurator URL) with the state's
// visual deltas applied. The user's orb identity is never replaced: a state
// reads through speed, intensity, and a rim/glow tint, never through a
// different style preset.

import { isSilentTool } from '@/lib/tool-render-class'

import { type OrbParams } from './orb-params'

/** Every assistant/thread state the orb can show. */
export type OrbState =
  | 'idle'
  | 'thinking'
  | 'streaming'
  | 'tool-running'
  | 'tool-result'
  | 'waiting-input'
  | 'model-loading'
  | 'compacting'
  | 'error'
  | 'cancelled'
  | 'complete'

export const orbStates: readonly OrbState[] = [
  'idle',
  'thinking',
  'streaming',
  'tool-running',
  'tool-result',
  'waiting-input',
  'model-loading',
  'compacting',
  'error',
  'cancelled',
  'complete'
]

/** What the turn's tail part is doing — derived from the last message part. */
export type OrbTailPhase = 'none' | 'reasoning' | 'streaming' | 'tool-running' | 'tool-result' | 'other'

/** Minimal structural shape `orbTailPhase` reads; both assistant-ui content
 * parts and the session-mirror `ChatMessagePart`s satisfy it. */
export interface OrbPartLike {
  type: string
  toolName?: string
  result?: unknown
  text?: unknown
}

/**
 * The turn's current phase from its tail part. A settled tool call stays the
 * tail until the next part arrives, so the gap after a result lands reads as
 * `tool-result`; silent tools (`todo`, reactions) narrate nothing and fall
 * through to `other`.
 */
export function orbTailPhase(parts: readonly OrbPartLike[]): OrbTailPhase {
  const last = parts[parts.length - 1]

  if (!last) {
    return 'none'
  }

  if (last.type === 'tool-call' && !isSilentTool(last.toolName ?? '')) {
    return last.result === undefined ? 'tool-running' : 'tool-result'
  }

  if (last.type === 'reasoning') {
    return 'reasoning'
  }

  if (last.type === 'text') {
    return typeof last.text === 'string' && last.text.length > 0 ? 'streaming' : 'other'
  }

  return 'other'
}

/**
 * Flat input bag for `resolveOrbState` — every signal the resolver needs,
 * nothing it doesn't. `useOrbState` projects the session stores onto this.
 */
export interface OrbStateSignals {
  /** assistant-ui message status type, or undefined when there is no message. */
  statusType?: string
  /** assistant-ui message status reason. */
  statusReason?: string
  /** the message carries a string error payload. */
  messageError: boolean
  /** what the turn's tail part is doing (only read while running/busy). */
  tailPhase: OrbTailPhase
  /** a tool call's arguments are still arriving (pre-call draft). */
  draftingTool: boolean
  /** the turn is paused on the user: clarify / approval / sudo / secret. */
  awaitingInput: boolean
  /** a provider wait frame is narrating (or the model is still loading). */
  providerWait: boolean
  /** auto-compaction ("Summarizing thread") owns the turn. */
  compacting: boolean
  /** the session-level turn-busy (covers gaps between message bubbles). */
  busy: boolean
  /** the turn just settled successfully — brief flash before `idle`. */
  justCompleted: boolean
}

/**
 * Resolve the orb state from flat signals. Precedence is deliberate:
 *
 * 1. `error` — a failed turn is the most important thing to show; nothing
 *    outranks it.
 * 2. `cancelled` — any `incomplete` stop without an error payload (cancelled,
 *    length, content-filter, …) reads as a stopped turn.
 * 3. `waiting-input` — the turn is paused on the user; the actionable state
 *    wins over background phases.
 * 4. `compacting` — rarer and slower than a draft; explains a transcript that
 *    looks like it reset (same ranking as the status hint).
 * 5. `model-loading` — a narrated provider wait / local model load.
 * 6. Running phases, from the tail part: `tool-running` (in flight or still
 *    being drafted), `tool-result` (result landed, turn still working),
 *    `streaming` (tokens flowing), otherwise `thinking`.
 * 7. `complete` — the brief settle flash after a successful turn.
 * 8. `idle` — nothing to show.
 */
export function resolveOrbState(s: OrbStateSignals): OrbState {
  if (s.messageError || (s.statusType === 'incomplete' && s.statusReason === 'error')) {
    return 'error'
  }

  if (s.statusType === 'incomplete') {
    return 'cancelled'
  }

  if (s.statusType === 'requires-action' || s.awaitingInput) {
    return 'waiting-input'
  }

  if (s.compacting) {
    return 'compacting'
  }

  if (s.providerWait) {
    return 'model-loading'
  }

  if (s.statusType === 'running' || s.busy) {
    if (s.tailPhase === 'tool-running' || s.draftingTool) {
      return 'tool-running'
    }

    if (s.tailPhase === 'tool-result') {
      return 'tool-result'
    }

    if (s.tailPhase === 'streaming') {
      return 'streaming'
    }

    return 'thinking'
  }

  if (s.justCompleted) {
    return 'complete'
  }

  return 'idle'
}

/**
 * The visual delta a state applies on top of the user's orb. Multipliers
 * compose with the base params (custom BYO URL or the built-in default);
 * `glow` tints only the rim/glow — never the orb's body colors — so a custom
 * orb keeps its identity while the state stays readable.
 */
export interface OrbStateVisual {
  /** What the state looks like — the documented mapping. */
  description: string
  /** Speed multiplier on the user's base speed. */
  speed: number
  /** Exposure (brightness/intensity) multiplier. */
  exposure: number
  /** Radius multiplier. */
  radius: number
  /** Rim/glow tint override; null keeps the user's colors. */
  glow: string | null
}

/**
 * State → visual mapping. Every state is a distinct combination of
 * speed / intensity / glow tint; the orb's style preset and body colors
 * always stay the user's own.
 */
export const orbStateVisuals: Record<OrbState, OrbStateVisual> = {
  idle: {
    description: 'Slow breathing at reduced size and brightness — the assistant is at rest.',
    speed: 0.35,
    exposure: 0.9,
    radius: 0.92,
    glow: null
  },
  thinking: {
    description: 'The base orb at full motion — the model is reasoning, no output yet.',
    speed: 1,
    exposure: 1,
    radius: 1,
    glow: null
  },
  streaming: {
    description: 'Fast and bright — response tokens are flowing.',
    speed: 1.6,
    exposure: 1.15,
    radius: 1,
    glow: null
  },
  'tool-running': {
    description: 'Brisk with a cyan rim — a tool call is in flight (or its arguments are still arriving).',
    speed: 1.3,
    exposure: 1.1,
    radius: 1,
    glow: '#38BDF8'
  },
  'tool-result': {
    description: 'Bright settle pulse, motion eased — a tool result landed and the turn is still working.',
    speed: 0.8,
    exposure: 1.3,
    radius: 1,
    glow: null
  },
  'waiting-input': {
    description: 'Slow with an amber rim — the turn is paused on the user (clarify, approval, sudo, secret).',
    speed: 0.5,
    exposure: 1,
    radius: 0.96,
    glow: '#F5B544'
  },
  'model-loading': {
    description: 'Dimmed and unhurried — a provider wait or local model load is narrating.',
    speed: 0.7,
    exposure: 0.85,
    radius: 0.95,
    glow: null
  },
  compacting: {
    description: 'Slow with a violet rim — auto-compaction ("Summarizing thread") owns the turn.',
    speed: 0.6,
    exposure: 1,
    radius: 0.96,
    glow: '#A78BFA'
  },
  error: {
    description: 'Slow with a red rim, slightly hot — the turn failed.',
    speed: 0.4,
    exposure: 1.2,
    radius: 0.96,
    glow: '#F87171'
  },
  cancelled: {
    description: 'Dimmed to gray and nearly still — the turn stopped before finishing.',
    speed: 0.3,
    exposure: 0.7,
    radius: 0.94,
    glow: '#9CA3AF'
  },
  complete: {
    description: 'Brief green-tinted settle flash after a successful turn, then back to idle.',
    speed: 0.5,
    exposure: 1.1,
    radius: 1,
    glow: '#34D399'
  }
}

/**
 * Render params for an orb state: the user's base params (custom BYO URL or
 * the built-in default) with the state's visual deltas applied.
 */
export function orbParamsForState(state: OrbState, base: OrbParams): OrbParams {
  const visual = orbStateVisuals[state]

  return {
    ...base,
    speed: base.speed * visual.speed,
    exposure: base.exposure * visual.exposure,
    radius: base.radius * visual.radius,
    ...(visual.glow === null
      ? {}
      : { glowColor: visual.glow, shellEdge: visual.glow })
  }
}
