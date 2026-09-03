import { describe, expect, it } from 'vitest'

import {
  getActiveSessionOpenPerfFixture,
  runSessionOpenPerfFixture,
  setSessionOpenPerfFixtureRunner
} from './session-open-perf-fixture.noop'

describe('shipped production session-open perf fixture stand-in', () => {
  it('cannot register or run a controlled fixture', async () => {
    setSessionOpenPerfFixtureRunner(async () => undefined)

    expect(getActiveSessionOpenPerfFixture()).toBeNull()
    await expect(
      runSessionOpenPerfFixture({
        delayRuntimeMs: 0,
        fetchLatest: async () => ({ messages: [], session_id: 'synthetic' }),
        requestGateway: async () => ({}),
        storedSessionId: 'synthetic'
      })
    ).rejects.toThrow('unavailable in a normal production renderer')
  })
})
