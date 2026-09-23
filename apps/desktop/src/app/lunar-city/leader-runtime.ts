import { normalizeProfileKey } from '@/store/profile'

import { type LeaderOwner, leaderOwnerKey } from './leader-sessions'
import type { EntityKey, LeaderAnimationState, LeaderId, LeaderStateClipMap, LunarEntity } from './model'

export interface LeaderConversationState {
  session: Pick<LeaderSessionState, 'awaitingResponse' | 'busy' | 'interrupted'>
  sessionAvailable: boolean
  voice: Pick<LeaderVoiceState, 'active' | 'status'>
}

export interface LeaderSessionState {
  awaitingResponse: boolean
  busy: boolean
  interrupted: boolean
}

export interface LeaderVoiceState {
  active: boolean
  status: 'idle' | 'listening' | 'speaking' | 'thinking' | 'transcribing'
}

export interface LeaderAnimationProjection {
  clip: string | undefined
  state: LeaderAnimationState
}

const LEADER_MODELS = ['owl', 'fox', 'badger', 'otter', 'bird', 'stag'] as const satisfies readonly LeaderId[]

/**
 * A conversation can only be opened for a profile entity whose source identity
 * is already present in the immutable city snapshot.  Session/worker picks and
 * display labels deliberately have no path into this resolver.
 */
export function leaderOwnerForProfile(entity: LunarEntity): LeaderOwner | undefined {
  if (entity.identity.kind !== 'profile') {
    return undefined
  }

  const connectionId = entity.identity.connectionId.trim()
  const profile = normalizeProfileKey(entity.identity.profile)

  return connectionId && profile ? { connectionId, profile } : undefined
}

/** A unique presentation identity, separate from the finite curated model set. */
export function leaderVisualIdForOwner(owner: LeaderOwner): string {
  return `lunar-city:leader:${leaderOwnerKey(owner)}`
}

/**
 * A profile-qualified camera anchor. The underlying GLB models stay shared
 * presentation assets; this key exists only so a conversation selection never
 * aliases a static animal pick or an identically named profile on another
 * connection.
 */
export function leaderFocusKeyForOwner(owner: LeaderOwner): EntityKey {
  return leaderVisualIdForOwner(owner) as EntityKey
}

/**
 * Profiles are the only live identities permitted to own a leader
 * conversation. Sorting by the encoded canonical owner key keeps the small
 * interaction rail stable across unrelated worker publications.
 */
export function profileLeaders(entities: ReadonlyMap<EntityKey, LunarEntity>): readonly LunarEntity[] {
  return [...entities.values()]
    .filter(entity => leaderOwnerForProfile(entity) !== undefined)
    .sort((left, right) =>
      leaderOwnerKey(leaderOwnerForProfile(left)!).localeCompare(leaderOwnerKey(leaderOwnerForProfile(right)!))
    )
}

/**
 * The character model is a deterministic presentation choice derived from the
 * full exact owner key. It never participates in session routing or mutation.
 */
export function leaderModelIdForOwner(owner: LeaderOwner, models: readonly LeaderId[] = LEADER_MODELS): LeaderId {
  const key = leaderOwnerKey(owner)
  let hash = 0

  for (const character of key) {
    hash = (hash * 31 + character.codePointAt(0)!) >>> 0
  }

  return models[hash % models.length] ?? LEADER_MODELS[0]
}

/**
 * Camera movement intentionally targets the shared model's declared anchor.
 * The caller retains the full owner identity for every session operation;
 * this presentation key cannot select, route, or mutate a leader by itself.
 */
export function leaderModelFocusKeyForOwner(owner: LeaderOwner, models?: readonly LeaderId[]): EntityKey {
  return `lunar-city:leader:${leaderModelIdForOwner(owner, models)}` as EntityKey
}

/**
 * Converts only observed standard-session and existing voice-hook lifecycle to
 * a GLB-declared clip. A click, RPC acknowledgement, or local pending state
 * cannot advance the visible leader state.
 */
export function leaderAnimationForConversation({
  clips,
  session,
  sessionAvailable,
  voice
}: LeaderConversationState & { clips: LeaderStateClipMap }): LeaderAnimationProjection {
  let state: LeaderAnimationState

  if (!sessionAvailable) {
    state = 'unavailable'
  } else if (voice.active && (voice.status === 'listening' || voice.status === 'transcribing')) {
    state = 'listening'
  } else if (voice.active && voice.status === 'speaking') {
    state = 'talking'
  } else if (voice.active && voice.status === 'thinking') {
    state = 'thinking'
  } else if (session.busy || session.awaitingResponse) {
    state = 'thinking'
  } else if (session.interrupted) {
    state = 'acknowledging'
  } else {
    state = 'idle'
  }

  return { clip: clips[state], state }
}
