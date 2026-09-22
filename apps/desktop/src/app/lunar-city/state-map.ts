import type { AuthorityState, DestinationId } from './model'

export interface ObservedState {
  source: 'kanban' | 'session' | 'subagent'
  status: string
  fresh: boolean
  authority?: AuthorityState
  connected?: boolean
}

export interface SpatialState {
  animation: string
  authority: AuthorityState
  destination: DestinationId
}

const UNKNOWN_STATE: SpatialState = Object.freeze({
  animation: 'unavailable',
  authority: 'unknown',
  destination: 'unknown'
})

const STALE_STATE: SpatialState = Object.freeze({
  animation: 'unavailable',
  authority: 'stale',
  destination: 'unknown'
})

const PARTIAL_STATE: SpatialState = Object.freeze({
  animation: 'unavailable',
  authority: 'partial',
  destination: 'unknown'
})

const AUTHORITATIVE_STATES: Readonly<Record<string, Omit<SpatialState, 'authority'>>> = Object.freeze({
  blocked: { animation: 'blocked', destination: 'triage' },
  completed: { animation: 'done', destination: 'project' },
  dependency: { animation: 'handoff', destination: 'council' },
  done: { animation: 'done', destination: 'project' },
  failed: { animation: 'failed', destination: 'triage' },
  heartbeat: { animation: 'heartbeat', destination: 'garden' },
  idle: { animation: 'rest', destination: 'garden' },
  orchestration: { animation: 'handoff', destination: 'council' },
  pause: { animation: 'rest', destination: 'garden' },
  paused: { animation: 'rest', destination: 'garden' },
  queued: { animation: 'queue', destination: 'bus' },
  ready: { animation: 'queue', destination: 'bus' },
  recovery: { animation: 'rest', destination: 'garden' },
  resource_wait: { animation: 'wait', destination: 'depot' },
  review: { animation: 'review', destination: 'review' },
  running: { animation: 'work', destination: 'project' },
  triage: { animation: 'triage', destination: 'triage' },
  waiting_for_resource: { animation: 'wait', destination: 'depot' },
  working: { animation: 'work', destination: 'project' }
})

/**
 * Turns an observed status into presentation-only movement.  It does not
 * infer work from a partial, stale, disconnected, or unknown source.
 */
export function mapObservedState(input: ObservedState): SpatialState {
  if (!input.fresh) {
    return STALE_STATE
  }

  if (input.connected === false || input.authority === 'unknown') {
    return UNKNOWN_STATE
  }

  if (input.authority === 'partial') {
    return PARTIAL_STATE
  }

  if (input.authority === 'stale') {
    return STALE_STATE
  }

  const state =
    AUTHORITATIVE_STATES[
      input.status
        .trim()
        .toLowerCase()
        .replace(/[\s-]+/gu, '_')
    ]

  return state ? { ...state, authority: 'authoritative' } : UNKNOWN_STATE
}
