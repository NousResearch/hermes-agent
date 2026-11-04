import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionTileDelegate } from './session-states'

const fakeDelegate = () => ({}) as unknown as SessionTileDelegate

describe('session tile delegate revision', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('wakes tiles when a delegate first arrives, not when it is replaced', async () => {
    // Every tile pane subscribes to the revision, and the wiring re-registers
    // its delegate whenever a callback changes identity (every sessions.changed
    // tick): a bump per replacement re-rendered every tile's chat shell.
    const { $sessionTileDelegateRevision, sessionTileDelegate, setSessionTileDelegate } =
      await import('./session-states')

    const notify = vi.fn()
    const stop = $sessionTileDelegateRevision.listen(notify)

    const first = fakeDelegate()
    setSessionTileDelegate(first)
    expect(notify).toHaveBeenCalledTimes(1)

    const replacement = fakeDelegate()
    setSessionTileDelegate(fakeDelegate())
    setSessionTileDelegate(replacement)
    expect(notify).toHaveBeenCalledTimes(1)
    // Actions read the delegate at call time, so they still get the latest.
    expect(sessionTileDelegate()).toBe(replacement)

    stop()
  })
})
