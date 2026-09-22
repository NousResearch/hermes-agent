import { leaderModelIdForOwner, leaderOwnerForProfile, profileLeaders } from '../leader-runtime'
import { leaderOwnerKey } from '../leader-sessions'
import type { LeaderId, LunarCitySnapshot } from '../model'

import type { LeaderLifeMode } from './leader-life'

export interface ObservedLeaderLife {
  id: LeaderId
  mode: LeaderLifeMode
  ownerKeys: readonly string[]
  reason: 'observed-work' | 'observed-idle' | 'no-session-observation' | 'stale-owner' | 'ambiguous-model'
}

/** Projects already observed session presentation, never creating work or routing a command.
 * Shared animal models cannot claim the activity of one arbitrarily chosen owner. */
export function observedLeaderLife(snapshot: LunarCitySnapshot, models: readonly LeaderId[]): readonly ObservedLeaderLife[] {
  const groups = new Map<LeaderId, ReturnType<typeof profileLeaders>[number][]>()

  for (const profile of profileLeaders(snapshot.entities)) {
    const id = leaderModelIdForOwner(leaderOwnerForProfile(profile)!, models)
    const owners = groups.get(id) ?? []
    owners.push(profile); groups.set(id, owners)
  }

  return [...groups].map(([id, profiles]) => {
    const ownerKeys = profiles.map(profile => leaderOwnerKey(leaderOwnerForProfile(profile)!))

    if (profiles.length !== 1) {return { id, ownerKeys, mode: 'unavailable', reason: 'ambiguous-model' }}
    const profile = profiles[0]!, owner = leaderOwnerForProfile(profile)!

    const sessions = [...snapshot.entities.values()].filter(entity =>
      (entity.identity.kind === 'session' || entity.identity.kind === 'subagent') &&
      entity.identity.connectionId === owner.connectionId && entity.identity.profile === owner.profile)

    if (profile.authority !== 'authoritative' || sessions.some(entity => entity.authority !== 'authoritative')) {
      return { id, ownerKeys, mode: 'unavailable', reason: 'stale-owner' }
    }

    if (!sessions.length) {return { id, ownerKeys, mode: 'home', reason: 'no-session-observation' }}

    // Session/subagent adapters map actual running observations to work. This is
    // only a visual projection of that existing state, never command authority.
    if (sessions.some(entity => entity.animation === 'work')) {return { id, ownerKeys, mode: 'work', reason: 'observed-work' }}

    return { id, ownerKeys, mode: 'idle', reason: 'observed-idle' }
  })
}
