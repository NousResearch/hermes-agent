import { afterEach, describe, expect, it } from 'vitest'

import { $lastViewedSchedule, pickSchedule, rememberViewedSchedule } from './landing'

const listed = [{ id: 'brief' }, { id: 'digest' }, { id: 'triage' }]

afterEach(() => $lastViewedSchedule.set(null))

describe('pickSchedule', () => {
  it('shows the schedule asked for while it is listed', () => {
    expect(pickSchedule(listed, 'digest')).toBe(listed[1])
  })

  it('falls back to the first listed schedule when the asked-for one was deleted or searched away', () => {
    expect(pickSchedule(listed, 'gone')).toBe(listed[0])
    expect(pickSchedule(listed.slice(2), 'digest')).toBe(listed[2])
  })

  it('shows the first schedule with nothing asked for, and nothing for an empty list', () => {
    expect(pickSchedule(listed, null)).toBe(listed[0])
    expect(pickSchedule([], 'brief')).toBeNull()
  })
})

describe('rememberViewedSchedule', () => {
  it('keeps the last schedule this window showed', () => {
    rememberViewedSchedule('brief')
    rememberViewedSchedule('triage')

    expect($lastViewedSchedule.get()).toBe('triage')
  })
})
