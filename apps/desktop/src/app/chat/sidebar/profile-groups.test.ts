import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { buildProfileGroups } from './profile-groups'

/** Build the smallest session row the profile grouper needs. */
const row = (id: string, profile?: string): SessionInfo =>
  ({ id, message_count: 1, profile, source: 'desktop', started_at: 0, title: id }) as SessionInfo

describe('buildProfileGroups', () => {
  it('groups sessions under their normalized profile and keeps default first without a focus', () => {
    const groups = buildProfileGroups(
      [row('r1', 'research'), row('d1', 'default'), row('a1', 'analyst'), row('r2', 'research')],
      {},
      null
    )

    expect(groups.map(group => group.id)).toEqual(['default', 'analyst', 'research'])
    expect(groups[2].sessions.map(session => session.id)).toEqual(['r1', 'r2'])
  })

  it('treats legacy rows without a profile as default', () => {
    const groups = buildProfileGroups([row('legacy'), row('a1', 'analyst')], {}, null)

    expect(groups.map(group => group.id)).toEqual(['default', 'analyst'])
    expect(groups[0].sessions.map(session => session.id)).toEqual(['legacy'])
  })

  it('floats the focused profile above default and the alphabetical rest', () => {
    const groups = buildProfileGroups(
      [row('d1', 'default'), row('a1', 'analyst'), row('r1', 'research')],
      {},
      'research'
    )

    expect(groups.map(group => group.id)).toEqual(['research', 'default', 'analyst'])
  })

  it('keeps default first when it is itself the focused profile', () => {
    const groups = buildProfileGroups([row('d1', 'default'), row('a1', 'analyst')], {}, 'default')

    expect(groups.map(group => group.id)).toEqual(['default', 'analyst'])
  })

  it('normalizes the focused profile before ranking', () => {
    const groups = buildProfileGroups([row('a1', 'analyst'), row('r1', 'research')], {}, '  research ')

    expect(groups.map(group => group.id)).toEqual(['research', 'analyst'])
  })

  it('ignores a focused profile that has no sessions', () => {
    const groups = buildProfileGroups([row('d1', 'default'), row('a1', 'analyst')], {}, 'ghost')

    expect(groups.map(group => group.id)).toEqual(['default', 'analyst'])
  })

  it('stamps profile colors on the headers and none on default', () => {
    const groups = buildProfileGroups([row('d1', 'default'), row('a1', 'analyst')], { analyst: '#123456' }, null)

    expect(groups.find(group => group.id === 'analyst')?.color).toBe('#123456')
    expect(groups.find(group => group.id === 'default')?.color).toBeNull()
  })
})
