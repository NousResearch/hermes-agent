/**
 * Rate breaker for raw fs.watch handles (#118974): a win32 event storm must
 * close the watcher and fall back to a slow snapshot poll.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'

import {
  guardedWatch,
  type GuardedWatchOptions,
  stormBreakerArmed,
  WATCH_STORM_MAX_EVENTS,
  WATCH_STORM_POLL_MS,
  WATCH_STORM_WINDOW_MS,
  type WatchListener
} from './watch-storm-breaker'

function fakeWatcher() {
  let listener: WatchListener | null = null
  const close = vi.fn()

  return {
    close,
    emit(times = 1, filename = 'x') {
      for (let i = 0; i < times; i += 1) {
        listener?.('rename', filename)
      }
    },
    watch: (l: WatchListener) => {
      listener = l

      return { close }
    }
  }
}

function setup(platform: string, overrides: Partial<GuardedWatchOptions> = {}) {
  const fw = fakeWatcher()
  let snap = 'a'
  const onEvent = vi.fn()
  const onPollChange = vi.fn()
  const onTrip = vi.fn()

  const guard = guardedWatch({
    platform,
    watch: fw.watch,
    onEvent,
    snapshot: () => snap,
    onPollChange,
    onTrip,
    ...overrides
  })

  return {
    fw,
    guard,
    onEvent,
    onPollChange,
    onTrip,
    setSnap: (s: string) => {
      snap = s
    }
  }
}

describe('guardedWatch', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  test('arms only on win32', () => {
    expect(stormBreakerArmed('win32')).toBe(true)
    expect(stormBreakerArmed('darwin')).toBe(false)
    expect(stormBreakerArmed('linux')).toBe(false)
  })

  test('forwards events under the threshold and stays on the watcher', () => {
    const { fw, guard, onEvent } = setup('win32')

    fw.emit(WATCH_STORM_MAX_EVENTS)

    expect(onEvent).toHaveBeenCalledTimes(WATCH_STORM_MAX_EVENTS)
    expect(fw.close).not.toHaveBeenCalled()
    expect(guard.mode()).toBe('watch')
  })

  test('the window resets, so a steady rate under the threshold never trips', () => {
    const { fw, guard } = setup('win32')

    for (let i = 0; i < 5; i += 1) {
      fw.emit(WATCH_STORM_MAX_EVENTS)
      vi.advanceTimersByTime(WATCH_STORM_WINDOW_MS)
    }

    expect(guard.mode()).toBe('watch')
    expect(fw.close).not.toHaveBeenCalled()
  })

  test('a win32 storm closes the watcher, drops further events, and polls', () => {
    const { fw, guard, onEvent, onPollChange, onTrip, setSnap } = setup('win32')

    fw.emit(WATCH_STORM_MAX_EVENTS + 50_000)

    expect(fw.close).toHaveBeenCalledTimes(1)
    expect(guard.mode()).toBe('poll')
    expect(onTrip).toHaveBeenCalledTimes(1)
    expect(onEvent).toHaveBeenCalledTimes(WATCH_STORM_MAX_EVENTS)

    // Unchanged snapshot: no change reported.
    vi.advanceTimersByTime(WATCH_STORM_POLL_MS * 3)
    expect(onPollChange).not.toHaveBeenCalled()

    // Changed snapshot: reported once per change on the next poll.
    setSnap('b')
    vi.advanceTimersByTime(WATCH_STORM_POLL_MS - 1)
    expect(onPollChange).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(onPollChange).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(WATCH_STORM_POLL_MS * 2)
    expect(onPollChange).toHaveBeenCalledTimes(1)
  })

  test('the same storm on other platforms is forwarded untouched', () => {
    const { fw, guard, onEvent, onTrip } = setup('darwin')

    fw.emit(WATCH_STORM_MAX_EVENTS * 3)

    expect(onEvent).toHaveBeenCalledTimes(WATCH_STORM_MAX_EVENTS * 3)
    expect(fw.close).not.toHaveBeenCalled()
    expect(onTrip).not.toHaveBeenCalled()
    expect(guard.mode()).toBe('watch')
  })

  test('close() stops the poll and is idempotent', () => {
    const { fw, guard, onPollChange, setSnap } = setup('win32')

    fw.emit(WATCH_STORM_MAX_EVENTS + 1)
    guard.close()
    guard.close()

    setSnap('b')
    vi.advanceTimersByTime(WATCH_STORM_POLL_MS * 3)

    expect(onPollChange).not.toHaveBeenCalled()
    expect(fw.close).toHaveBeenCalledTimes(1)
    expect(guard.mode()).toBe('closed')
    expect(vi.getTimerCount()).toBe(0)
  })

  test('close() before a trip closes the watcher once', () => {
    const { fw, guard } = setup('win32')

    guard.close()

    expect(fw.close).toHaveBeenCalledTimes(1)
    expect(guard.mode()).toBe('closed')
  })

  test('a throwing snapshot (target vanished) is treated as a change, not a crash', () => {
    let gone = false

    const { fw, onPollChange } = setup('win32', {
      snapshot: () => {
        if (gone) {
          throw new Error('ENOENT')
        }

        return 'a'
      }
    })

    fw.emit(WATCH_STORM_MAX_EVENTS + 1)
    gone = true
    vi.advanceTimersByTime(WATCH_STORM_POLL_MS)

    expect(onPollChange).toHaveBeenCalledTimes(1)
  })

  test('an empty directory vanishing is still a change (error is not the empty snapshot)', () => {
    let gone = false

    const { fw, onPollChange } = setup('win32', {
      snapshot: () => {
        if (gone) {
          throw new Error('ENOENT')
        }

        return ''
      }
    })

    fw.emit(WATCH_STORM_MAX_EVENTS + 1)
    gone = true
    vi.advanceTimersByTime(WATCH_STORM_POLL_MS)

    expect(onPollChange).toHaveBeenCalledTimes(1)
  })
})
