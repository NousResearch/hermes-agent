import { describe, expect, it, vi } from 'vitest'

import {
  createKeepAwake,
  KEEP_AWAKE_RELEASE_GRACE_MS,
  keepAwakeHeld,
  keepAwakeWanted,
  parseKeepAwakeMode,
  type PowerSaveBlockerLike,
  readKeepAwakeMode
} from './power-save'

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
})

describe('keep-awake mode', () => {
  it('parses the three modes and the boolean the pre-mode toggle sent', () => {
    expect(parseKeepAwakeMode('off')).toBe('off')
    expect(parseKeepAwakeMode('while-working')).toBe('while-working')
    expect(parseKeepAwakeMode('always')).toBe('always')
    // The old Switch sent a boolean; on meant what 'always' means now.
    expect(parseKeepAwakeMode(true)).toBe('always')
    expect(parseKeepAwakeMode(false)).toBe('off')
    // Anything else is refused rather than guessed.
    expect(parseKeepAwakeMode('on')).toBeNull()
    expect(parseKeepAwakeMode(1)).toBeNull()
    expect(parseKeepAwakeMode(undefined)).toBeNull()
  })

  it('reads the persisted copy in the new and the legacy shape', () => {
    expect(readKeepAwakeMode({ mode: 'while-working' })).toBe('while-working')
    // keep-awake.json written before modes existed.
    expect(readKeepAwakeMode({ on: true })).toBe('always')
    expect(readKeepAwakeMode({ on: false })).toBe('off')
    // An explicit mode wins over a stale legacy flag beside it.
    expect(readKeepAwakeMode({ mode: 'off', on: true })).toBe('off')
    // Unreadable or absent -> off, the pre-existing default.
    expect(readKeepAwakeMode(null)).toBe('off')
    expect(readKeepAwakeMode('garbage')).toBe('off')
    expect(readKeepAwakeMode({ mode: 'sometimes' })).toBe('off')
  })

  it('holds the blocker only when the mode and the live turn picture agree', () => {
    expect(keepAwakeWanted('off', true)).toBe(false)
    expect(keepAwakeWanted('off', false)).toBe(false)
    expect(keepAwakeWanted('always', true)).toBe(true)
    expect(keepAwakeWanted('always', false)).toBe(true)
    // The point of the mode: follows the turn in and out.
    expect(keepAwakeWanted('while-working', true)).toBe(true)
    expect(keepAwakeWanted('while-working', false)).toBe(false)
  })

  it('drives the real blocker in and out of a turn under while-working', () => {
    const { blocker } = fakeBlocker()
    const keepAwake = createKeepAwake(blocker)
    const apply = (working: boolean) => keepAwake.set(keepAwakeWanted('while-working', working))

    apply(false)
    expect(blocker.start).not.toHaveBeenCalled()
    apply(true)
    apply(true) // a second active-work report mid-turn must not stack blockers
    expect(blocker.start).toHaveBeenCalledTimes(1)
    apply(false)
    expect(blocker.stop).toHaveBeenCalledTimes(1)
    expect(keepAwake.isActive()).toBe(false)
  })
})

describe('release grace (#134434)', () => {
  it('is a hold long enough to bridge a cold follow-up turn accept', () => {
    // The gap it bridges: turn end -> process_complete prompt -> model request
    // -> message.start re-arm. Five minutes dwarfs the observed 1-2 s release
    // windows and the tens of seconds a cold model request can take.
    expect(KEEP_AWAKE_RELEASE_GRACE_MS).toBeGreaterThanOrEqual(5 * 60_000)
  })

  it('keeps the blocker held through the grace window after the picture goes idle', () => {
    // The regression: a bare working=false used to release immediately.
    expect(keepAwakeHeld('while-working', false, false)).toBe(false)
    expect(keepAwakeHeld('while-working', false, true)).toBe(true)
  })

  it('holds during live work regardless of the grace state', () => {
    expect(keepAwakeHeld('while-working', true, false)).toBe(true)
    expect(keepAwakeHeld('while-working', true, true)).toBe(true)
  })

  it('never applies to off or always', () => {
    expect(keepAwakeHeld('off', false, true)).toBe(false)
    expect(keepAwakeHeld('off', true, true)).toBe(false)
    expect(keepAwakeHeld('always', false, false)).toBe(true)
    expect(keepAwakeHeld('always', false, true)).toBe(true)
  })
})
