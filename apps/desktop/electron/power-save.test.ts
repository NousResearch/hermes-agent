import { describe, expect, it, vi } from 'vitest'

import { createKeepAwake, type PowerSaveBlockerLike, shouldHoldAwake } from './power-save'

function fakeBlocker() {
  let next = 1
  const started = new Set<number>()

  const blocker: PowerSaveBlockerLike = {
    isStarted: id => started.has(id),
    start: vi.fn(() => {
      const id = next++
      started.add(id)

      return id
    }),
    stop: vi.fn(id => void started.delete(id))
  }

  return { blocker, started }
}

describe('createKeepAwake', () => {
  it('starts once, is idempotent, and stops', () => {
    const { blocker } = fakeBlocker()
    const keepAwake = createKeepAwake(blocker)

    expect(keepAwake.isActive()).toBe(false)
    expect(keepAwake.set(true)).toBe(true)
    keepAwake.set(true) // idempotent — no second blocker
    expect(blocker.start).toHaveBeenCalledTimes(1)
    expect(blocker.start).toHaveBeenCalledWith('prevent-app-suspension')

    expect(keepAwake.set(false)).toBe(false)
    keepAwake.set(false)
    expect(blocker.stop).toHaveBeenCalledTimes(1)
  })

  it('re-arms after the OS dropped the blocker', () => {
    const { blocker, started } = fakeBlocker()
    const keepAwake = createKeepAwake(blocker)

    keepAwake.set(true)
    started.clear() // system released it out from under us
    expect(keepAwake.isActive()).toBe(false)

    keepAwake.set(true)
    expect(blocker.start).toHaveBeenCalledTimes(2)
    expect(keepAwake.isActive()).toBe(true)
  })

  it('honors a custom blocker type', () => {
    const { blocker } = fakeBlocker()
    createKeepAwake(blocker, 'prevent-display-sleep').set(true)

    expect(blocker.start).toHaveBeenCalledWith('prevent-display-sleep')
  })
})

describe('shouldHoldAwake', () => {
  it('never holds while the preference is off', () => {
    expect(shouldHoldAwake(false, 0)).toBe(false)
    expect(shouldHoldAwake(false, 3)).toBe(false)
  })

  it('holds only while a turn is in flight', () => {
    expect(shouldHoldAwake(true, 0)).toBe(false)
    expect(shouldHoldAwake(true, 1)).toBe(true)
  })

  it('drives the blocker through a run lifecycle', () => {
    const { blocker } = fakeBlocker()
    const keepAwake = createKeepAwake(blocker)
    const apply = (turns: number) => keepAwake.set(shouldHoldAwake(true, turns))

    expect(apply(0)).toBe(false) // preference on, nothing running: let it sleep
    expect(apply(1)).toBe(true) // run starts
    expect(apply(2)).toBe(true) // second window joins — still one blocker
    expect(blocker.start).toHaveBeenCalledTimes(1)
    expect(apply(0)).toBe(false) // last run ends: released immediately
    expect(blocker.stop).toHaveBeenCalledTimes(1)
  })
})
