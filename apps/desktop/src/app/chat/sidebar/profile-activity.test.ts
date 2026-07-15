import { describe, expect, it } from 'vitest'

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
      workingSessionIds: ['default-run', 'needs-answer']
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
      workingSessionIds: ['running', 'blocked']
    })

    expect(activity.claire).toBe('needs-input')
  })

  it('keeps an unseen result visible while another session in that profile is running', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: [],
      sessions: [session('finished', 'claire'), session('running', 'claire')],
      unreadSessionIds: ['finished'],
      workingSessionIds: ['running']
    })

    expect(activity.claire).toBe('unread')
  })

  it('keeps one conversation working while its independent review follows an unseen parent completion', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: [],
      sessions: [session('parent', 'claire')],
      unreadSessionIds: ['parent'],
      workingSessionIds: ['parent']
    })

    expect(activity.claire).toBe('working')
  })

  it('matches activity through a compression lineage root and ignores unknown ids', () => {
    const activity = deriveProfileActivityByProfile({
      attentionSessionIds: ['missing-attention'],
      sessions: [session('live-tip', 'claire', 'lineage-root')],
      unreadSessionIds: ['missing-unread'],
      workingSessionIds: ['lineage-root']
    })

    expect(activity).toEqual({ claire: 'working' })
  })
})
