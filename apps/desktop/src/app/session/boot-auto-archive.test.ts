import { describe, expect, it, vi } from 'vitest'

import { type BootAutoArchiveState, scheduleBootAutoArchiveOnce } from './boot-auto-archive'

describe('scheduleBootAutoArchiveOnce', () => {
  it('defers boot maintenance so session loading can paint first', async () => {
    const scheduled: Array<() => void> = []
    const state: BootAutoArchiveState = { started: false }
    const autoArchive = vi.fn().mockResolvedValue({ archived: 0, ok: true })
    const onArchived = vi.fn()

    const scheduledNow = scheduleBootAutoArchiveOnce({
      autoArchive,
      onArchived,
      preserveIds: ['active', 'active', '', 'pinned'],
      schedule: callback => scheduled.push(callback),
      state
    })

    expect(scheduledNow).toBe(true)
    expect(state.started).toBe(true)
    expect(autoArchive).not.toHaveBeenCalled()
    expect(scheduled).toHaveLength(1)

    scheduled[0]()
    await Promise.resolve()

    expect(autoArchive).toHaveBeenCalledWith(['active', 'pinned'])
    expect(onArchived).not.toHaveBeenCalled()
  })

  it('runs only once and refreshes when boot maintenance archived rows', async () => {
    const scheduled: Array<() => void> = []
    const state: BootAutoArchiveState = { started: false }
    const autoArchive = vi.fn().mockResolvedValue({ archived: 3, ok: true })
    const onArchived = vi.fn()

    expect(
      scheduleBootAutoArchiveOnce({
        autoArchive,
        onArchived,
        preserveIds: ['a'],
        schedule: callback => scheduled.push(callback),
        state
      })
    ).toBe(true)

    expect(
      scheduleBootAutoArchiveOnce({
        autoArchive,
        onArchived,
        preserveIds: ['b'],
        schedule: callback => scheduled.push(callback),
        state
      })
    ).toBe(false)

    expect(scheduled).toHaveLength(1)

    scheduled[0]()
    await Promise.resolve()

    expect(autoArchive).toHaveBeenCalledTimes(1)
    expect(autoArchive).toHaveBeenCalledWith(['a'])
    expect(onArchived).toHaveBeenCalledWith({ archived: 3, ok: true })
  })

  it('reports errors without throwing from the scheduled callback', async () => {
    const scheduled: Array<() => void> = []
    const state: BootAutoArchiveState = { started: false }
    const error = new Error('offline')
    const autoArchive = vi.fn().mockRejectedValue(error)
    const onError = vi.fn()

    scheduleBootAutoArchiveOnce({
      autoArchive,
      onArchived: vi.fn(),
      onError,
      preserveIds: [],
      schedule: callback => scheduled.push(callback),
      state
    })

    scheduled[0]()
    await Promise.resolve()
    await Promise.resolve()

    expect(onError).toHaveBeenCalledWith(error)
  })
})
