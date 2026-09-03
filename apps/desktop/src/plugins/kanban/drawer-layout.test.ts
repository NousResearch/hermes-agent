import { describe, expect, it } from 'vitest'

import { drawerTabStats } from './drawer-layout'

describe('drawer tab stats', () => {
  it('counts human comments and machine events together in the timeline', () => {
    expect(
      drawerTabStats({
        comments: 3,
        events: 8,
        hasLog: false,
        running: false,
        runs: 2
      }).timelineCount
    ).toBe(11)
  })

  it('marks execution live only while the task is running', () => {
    expect(drawerTabStats({ comments: 0, events: 0, hasLog: true, running: true, runs: 1 }).executionLive).toBe(true)
    expect(drawerTabStats({ comments: 0, events: 0, hasLog: true, running: false, runs: 1 }).executionLive).toBe(false)
  })

  it('keeps execution discoverable when a log exists without run history', () => {
    expect(drawerTabStats({ comments: 0, events: 0, hasLog: true, running: false, runs: 0 }).executionCount).toBe(1)
  })

  it('uses actual run history as the execution count when present', () => {
    expect(drawerTabStats({ comments: 0, events: 0, hasLog: true, running: false, runs: 4 }).executionCount).toBe(4)
  })
})
