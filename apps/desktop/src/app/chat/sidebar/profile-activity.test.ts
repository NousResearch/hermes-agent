import { describe, expect, it } from 'vitest'

import { sessionScopeKey } from '@/store/session'
import { sessionActivityKey } from '@/store/session-activity'

import { deriveProfileActivityByProfile, type ProfileActivitySession } from './profile-activity'

const session = (id: string, profile?: string, lineageRootId?: string): ProfileActivitySession => ({
  id,
  profile,
  _lineage_root_id: lineageRootId
})

describe('deriveProfileActivityByProfile', () => {
  it('maps live session ids to their owning profiles', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: ['needs-answer'],
      sessions: [session('default-run'), session('needs-answer', 'claire'), session('finished', 'wallace')],
      unreadSessionIds: ['finished'],
      workingSessionIds: [sessionActivityKey('default', 'default-run'), sessionActivityKey('claire', 'needs-answer')]
    })

    expect(activity).toEqual({
      claire: 'needs-input',
      default: 'working',
      wallace: 'unread'
    })
  })

  it('lets the most actionable state win when one profile has several sessions', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: ['blocked'],
      sessions: [session('finished', 'claire'), session('running', 'claire'), session('blocked', 'claire')],
      unreadSessionIds: ['finished'],
      workingSessionIds: [sessionActivityKey('claire', 'running'), sessionActivityKey('claire', 'blocked')]
    })

    expect(activity.claire).toBe('needs-input')
  })

  it('keeps an unseen result visible while another session in that profile is running', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: [],
      sessions: [session('finished', 'claire'), session('running', 'claire')],
      unreadSessionIds: ['finished'],
      workingSessionIds: [sessionActivityKey('claire', 'running')]
    })

    expect(activity.claire).toBe('unread')
  })

  it('keeps an unseen parent completion above its still-running independent review', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: [],
      sessions: [session('parent', 'claire')],
      unreadSessionIds: ['parent'],
      workingSessionIds: [sessionActivityKey('claire', 'parent')]
    })

    expect(activity.claire).toBe('unread')
  })

  it('matches activity through a compression lineage root and ignores unknown ids', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: ['missing-attention'],
      sessions: [session('live-tip', 'claire', 'lineage-root')],
      unreadSessionIds: ['missing-unread'],
      workingSessionIds: [sessionActivityKey('claire', 'lineage-root')]
    })

    expect(activity).toEqual({ claire: 'working' })
  })

  it('does not project a same-id working session onto another profile', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: [],
      sessions: [session('same', 'alpha'), session('same', 'beta')],
      unreadSessionIds: [],
      workingSessionIds: [sessionActivityKey('alpha', 'same')]
    })

    expect(activity).toEqual({ alpha: 'working' })
  })

  it('does not let a raw compatibility id override scoped unread ownership', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: [],
      sessions: [session('same', 'alpha'), session('same', 'beta')],
      unreadSessionIds: ['same', sessionScopeKey('alpha', 'same')],
      workingSessionIds: []
    })

    expect(activity).toEqual({ alpha: 'unread' })
  })
})
