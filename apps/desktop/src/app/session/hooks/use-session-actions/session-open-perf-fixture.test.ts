import { describe, expect, it } from 'vitest'

import {
  getActiveSessionOpenPerfFixture,
  isCompletedSessionOpenPerfFixture,
  runSessionOpenPerfFixture,
  setSessionOpenPerfFixtureRunner
} from './session-open-perf-fixture'

describe('session-open perf fixture seam', () => {
  it('exposes the fixture only while it invokes the registered real resume runner', async () => {
    const fixture = {
      delayRuntimeMs: 2000,
      fetchLatest: async () => ({ messages: [], session_id: 'perf-session' }),
      requestGateway: async () => ({}) as never,
      storedSessionId: 'perf-session'
    }

    let observed = null

    setSessionOpenPerfFixtureRunner(async () => {
      observed = getActiveSessionOpenPerfFixture()
    })

    await runSessionOpenPerfFixture(fixture)

    expect(observed).toBe(fixture)
    expect(getActiveSessionOpenPerfFixture()).toBeNull()
    expect(isCompletedSessionOpenPerfFixture('perf-session')).toBe(true)
  })

  it('returns the exact runner cleanup after the active fixture scope closes', async () => {
    let cleaned = false

    setSessionOpenPerfFixtureRunner(async () => () => {
      cleaned = true
    })

    const cleanup = await runSessionOpenPerfFixture({
      delayRuntimeMs: 0,
      fetchLatest: async () => ({ messages: [], session_id: 'perf-session' }),
      requestGateway: async () => ({}) as never,
      storedSessionId: 'perf-session'
    })

    expect(getActiveSessionOpenPerfFixture()).toBeNull()
    expect(cleaned).toBe(false)
    cleanup()
    expect(cleaned).toBe(true)
  })
})
