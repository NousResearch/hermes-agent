import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopUpdateStage } from '@/global'

import { BUSY_STAGE_STALL_MS, isBusyUpdateStage, watchUpdateStageDeadlines } from './update-stage-deadlines'

interface State {
  applying: boolean
  stage: DesktopUpdateStage
  message: string
  percent: number | null
  log: readonly unknown[]
}

const busy = (stage: DesktopUpdateStage, message = 'working'): State => ({
  applying: true,
  stage,
  message,
  percent: null,
  log: []
})

describe('update stage deadlines', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('bounds every stage that can spin, and no settled one', () => {
    const spinning: DesktopUpdateStage[] = [
      'prepare',
      'fetch',
      'pull',
      'pydeps',
      'update',
      'rebuild',
      'restart',
      'done'
    ]
    const settled: DesktopUpdateStage[] = ['idle', 'manual', 'guiSkew', 'error']

    expect(spinning.every(isBusyUpdateStage)).toBe(true)
    expect(settled.some(isBusyUpdateStage)).toBe(false)
  })

  it('fires once a busy stage goes without progress for its deadline', () => {
    const store = atom<State>(busy('rebuild'))
    const onStall = vi.fn()
    const stop = watchUpdateStageDeadlines(store, onStall)

    vi.advanceTimersByTime(BUSY_STAGE_STALL_MS.rebuild - 1)
    expect(onStall).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(onStall).toHaveBeenCalledWith('rebuild')
    stop()
  })

  it('never fires while progress keeps arriving, however long the stage runs', () => {
    const store = atom<State>(busy('update'))
    const onStall = vi.fn()
    const stop = watchUpdateStageDeadlines(store, onStall)

    for (let line = 0; line < 10; line += 1) {
      vi.advanceTimersByTime(BUSY_STAGE_STALL_MS.update - 1000)
      store.set({ ...store.get(), log: [...store.get().log, line] })
    }

    expect(onStall).not.toHaveBeenCalled()
    stop()
  })

  it('keeps counting new log entries once the log is at its cap', () => {
    const entry = (): { message: string } => ({ message: 'still building' })
    const capped = Array.from({ length: 50 }, entry)
    const store = atom<State>({ ...busy('rebuild'), log: capped })
    const onStall = vi.fn()
    const stop = watchUpdateStageDeadlines(store, onStall)

    for (let report = 0; report < 10; report += 1) {
      vi.advanceTimersByTime(BUSY_STAGE_STALL_MS.rebuild - 1000)
      // ingestProgress's shape: same stage, message and percent; one new entry, oldest dropped.
      store.set({ ...store.get(), log: [...store.get().log, entry()].slice(-50) })
    }

    expect(onStall).not.toHaveBeenCalled()
    stop()
  })

  it('a settled stage disarms the timer', () => {
    const store = atom<State>(busy('fetch'))
    const onStall = vi.fn()
    const stop = watchUpdateStageDeadlines(store, onStall)

    store.set({ ...busy('error'), applying: false })
    vi.advanceTimersByTime(BUSY_STAGE_STALL_MS.fetch * 2)
    expect(onStall).not.toHaveBeenCalled()
    stop()
  })

  it('a handed-off restart that never quits the app is bounded too', () => {
    const store = atom<State>({ ...busy('restart'), applying: false })
    const onStall = vi.fn()
    const stop = watchUpdateStageDeadlines(store, onStall)

    vi.advanceTimersByTime(BUSY_STAGE_STALL_MS.restart)
    expect(onStall).toHaveBeenCalledWith('restart')
    stop()
  })
})
