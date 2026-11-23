import { afterEach, describe, expect, it } from 'vitest'

import { $toolProgress, setToolProgress } from './tool-progress'

describe('tool progress publishing', () => {
  afterEach(() => $toolProgress.set({}))

  it('does not notify subscribers when the reading is unchanged', () => {
    $toolProgress.set({})
    let notifications = 0
    const stop = $toolProgress.listen(() => (notifications += 1))

    setToolProgress('session', 'call', { message: 'downloading', percent: 40 })
    setToolProgress('session', 'call', { message: 'downloading', percent: 40 })
    setToolProgress('session', 'call', { message: 'downloading', percent: 41 })

    stop()
    expect(notifications).toBe(2)
  })
})
