import { afterEach, describe, expect, it, vi } from 'vitest'

import { $focusFollowsMouse, setFocusFollowsMouse } from '@/store/focus-follows-mouse'

import { startUiPreferencesSync, type UiPreferencesSnapshot } from './sync'

const KEY = 'hermes.desktop.focusFollowsMouse'

let stop = () => {}

afterEach(() => {
  stop()
  setFocusFollowsMouse(false)
  window.localStorage.removeItem(KEY)
  vi.useRealTimers()
})

describe('focus preference in the machine UI store', () => {
  it('adopts the machine value and publishes later user choices through the shared bridge', async () => {
    vi.useFakeTimers()

    let current: UiPreferencesSnapshot = {
      exists: true,
      generation: 'focus-test',
      revision: 1,
      values: { [KEY]: 'true' }
    }

    const writes: unknown[] = []
    let changed: ((snapshot: UiPreferencesSnapshot) => void) | undefined
    stop = startUiPreferencesSync({
      read: async () => current,
      write: async payload => {
        writes.push(payload)
        current = {
          ...current,
          revision: current.revision + 1,
          values: { ...current.values, [KEY]: payload.values[KEY]! }
        }

        return current
      },
      onChanged: callback => {
        changed = callback

        return () => {
          changed = undefined
        }
      }
    })
    await vi.runAllTimersAsync()
    expect($focusFollowsMouse.get()).toBe(true)
    setFocusFollowsMouse(false)
    await vi.runAllTimersAsync()
    expect(writes).toContainEqual({ values: { [KEY]: 'false' } })
    current = { ...current, revision: current.revision + 1, values: { [KEY]: 'true' } }
    changed!(current)
    await vi.runAllTimersAsync()
    expect($focusFollowsMouse.get()).toBe(true)
  })
})
