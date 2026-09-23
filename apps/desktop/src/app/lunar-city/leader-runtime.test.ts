import { describe, expect, it } from 'vitest'

import { entityKey } from './identity'
import {
  leaderAnimationForConversation,
  leaderFocusKeyForOwner,
  leaderOwnerForProfile,
  leaderVisualIdForOwner,
  profileLeaders
} from './leader-runtime'
import type { LeaderStateClipMap, LunarEntity } from './model'

const owl = (connectionId: string, profile: string): LunarEntity => {
  const identity = { connectionId, kind: 'profile' as const, profile }

  return {
    animation: 'rest',
    authority: 'authoritative',
    destination: 'garden',
    identity,
    key: entityKey(identity),
    observedAt: 42
  }
}

const clips: LeaderStateClipMap = {
  acknowledging: 'leader:owl:acknowledging',
  idle: 'leader:owl:idle',
  listening: 'leader:owl:listening',
  talking: 'leader:owl:talking',
  thinking: 'leader:owl:thinking',
  unavailable: 'leader:owl:unavailable'
}

describe('Lunar City leader runtime identity and lifecycle projection', () => {
  it('derives a conversation owner only from an exact live profile identity', () => {
    expect(leaderOwnerForProfile(owl('source-a', 'owl'))).toEqual({ connectionId: 'source-a', profile: 'owl' })
    expect(
      leaderOwnerForProfile({
        ...owl('source-a', 'owl'),
        identity: { connectionId: 'source-a', kind: 'session', profile: 'owl', sessionId: 's1' }
      })
    ).toBeUndefined()
  })

  it('keeps duplicate profile names on different connections bound to different deterministic visual assignments', () => {
    const first = leaderOwnerForProfile(owl('source-a', 'owl'))!
    const second = leaderOwnerForProfile(owl('source-b', 'owl'))!

    expect(leaderVisualIdForOwner(first)).not.toBe(leaderVisualIdForOwner(second))
    expect(leaderFocusKeyForOwner(first)).not.toBe(leaderFocusKeyForOwner(second))
  })

  it('exposes only profile-owned leader candidates in exact stable owner order', () => {
    const sourceBOwl = owl('source-b', 'owl')
    const sourceAOwl = owl('source-a', 'owl')

    const session = {
      ...owl('source-a', 'owl'),
      identity: { connectionId: 'source-a', kind: 'session' as const, profile: 'owl', sessionId: 's1' }
    }

    expect(
      profileLeaders(
        new Map([
          [sourceBOwl.key, sourceBOwl],
          [session.key, session],
          [sourceAOwl.key, sourceAOwl]
        ])
      )
    ).toEqual([sourceAOwl, sourceBOwl])
  })

  it('projects only authoritative session and voice lifecycle onto declared clips', () => {
    expect(
      leaderAnimationForConversation({
        clips,
        session: { awaitingResponse: false, busy: false, interrupted: false },
        sessionAvailable: true,
        voice: { active: false, status: 'idle' }
      })
    ).toEqual({ state: 'idle', clip: 'leader:owl:idle' })
    expect(
      leaderAnimationForConversation({
        clips,
        session: { awaitingResponse: true, busy: true, interrupted: false },
        sessionAvailable: true,
        voice: { active: false, status: 'idle' }
      })
    ).toEqual({ state: 'thinking', clip: 'leader:owl:thinking' })
    expect(
      leaderAnimationForConversation({
        clips,
        session: { awaitingResponse: false, busy: false, interrupted: false },
        sessionAvailable: true,
        voice: { active: true, status: 'speaking' }
      })
    ).toEqual({ state: 'talking', clip: 'leader:owl:talking' })
    expect(
      leaderAnimationForConversation({
        clips,
        session: { awaitingResponse: false, busy: false, interrupted: false },
        sessionAvailable: false,
        voice: { active: false, status: 'idle' }
      })
    ).toEqual({ state: 'unavailable', clip: 'leader:owl:unavailable' })
  })
})
