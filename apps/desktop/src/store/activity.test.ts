import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { buildRailTasks } from './activity'

describe('buildRailTasks profile identity', () => {
  it('joins a working identity to the matching profile row when stored ids collide', () => {
    const sessions = [
      { id: 'same-id', last_active: 1, profile: 'default', title: 'Default row' },
      { id: 'same-id', last_active: 2, profile: 'work', title: 'Work row' }
    ] as SessionInfo[]

    const tasks = buildRailTasks([{ profile: 'work', sessionId: 'same-id' }], sessions, null, {})

    expect(tasks).toHaveLength(1)
    expect(tasks[0].label).toBe('Work row')
  })
})
