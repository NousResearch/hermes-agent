import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { buildProfileGroups } from './profile-groups'

/** Build the smallest session row needed by the profile-groups tests. */
const row = (id: string, profile?: string): SessionInfo =>
  ({ id, message_count: 1, profile, source: 'signal', started_at: 0, title: id }) as SessionInfo

const NO_COLORS = {}

describe('buildProfileGroups', () => {
  it('groups sessions by profile key', () => {
    const groups = buildProfileGroups(
      [row('default-row', 'default'), row('work-row', 'work'), row('other-row', 'work')],
      NO_COLORS
    )

    expect(groups.map(group => [group.id, group.mode, group.sessions.length])).toEqual([
      ['default', 'profile', 1],
      ['work', 'profile', 2]
    ])
  })

  it('floats the default profile first, then sorts the rest alphabetically', () => {
    const groups = buildProfileGroups(
      [row('zeta-row', 'zeta'), row('alpha-row', 'alpha'), row('default-row', 'default')],
      NO_COLORS
    )

    expect(groups.map(group => group.id)).toEqual(['default', 'alpha', 'zeta'])
  })

  it('treats legacy rows without a profile as default', () => {
    const groups = buildProfileGroups([row('legacy-row'), row('work-row', 'work')], NO_COLORS)

    expect(groups.map(group => group.id)).toEqual(['default', 'work'])
    expect(groups[0].sessions.map(session => session.id)).toEqual(['legacy-row'])
  })

  it('carries the profile color from the overrides map', () => {
    const groups = buildProfileGroups([row('work-row', 'work')], { work: '#aabbcc' })

    expect(groups[0].color).toBe('#aabbcc')
  })

  it('keeps the default profile colorless', () => {
    const groups = buildProfileGroups([row('default-row', 'default')], { default: '#aabbcc' })

    expect(groups[0].color).toBeNull()
  })
})
