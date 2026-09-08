import { afterEach, describe, expect, it, vi } from 'vitest'

import { buildRailTasks } from './activity'

afterEach(() => vi.useRealTimers())

describe('buildRailTasks', () => {
  it.each([
    ['an unresolved running session', () => buildRailTasks(['session-a'], [], null, {})],
    [
      'a preview restart',
      () => buildRailTasks([], [], { status: 'running', taskId: 'preview-a', url: 'http://localhost:3000' }, {})
    ]
  ])('keeps %s stable when inputs have not changed', (_label, build) => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-08T00:00:00Z'))
    const first = build()

    vi.setSystemTime(new Date('2026-09-08T00:01:00Z'))
    const second = build()

    expect(second).toEqual(first)
  })
})
