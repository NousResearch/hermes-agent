// @vitest-environment node
import { describe, expect, it } from 'vitest'

import { DEFAULT_IDLE_HANGUP_SECONDS, shouldIdleHangupLiveVoice } from '@/lib/voice-live'

describe('GPT-Live idle hangup', () => {
  it('hangs up once quiet time reaches the configured deadline', () => {
    expect(
      shouldIdleHangupLiveVoice({
        delegationInFlight: false,
        idleHangupSeconds: 300,
        quietSeconds: 300
      })
    ).toBe(true)
  })

  it('does not hang up before the deadline', () => {
    expect(
      shouldIdleHangupLiveVoice({
        delegationInFlight: false,
        idleHangupSeconds: 300,
        quietSeconds: 299
      })
    ).toBe(false)
  })

  it('never hangs up when the deadline is 0', () => {
    expect(
      shouldIdleHangupLiveVoice({
        delegationInFlight: false,
        idleHangupSeconds: 0,
        quietSeconds: 10_000
      })
    ).toBe(false)
  })

  it('does not hang up while a Hermes delegation is in flight', () => {
    expect(
      shouldIdleHangupLiveVoice({
        delegationInFlight: true,
        idleHangupSeconds: 300,
        quietSeconds: 600
      })
    ).toBe(false)
  })

  it('defaults to a five-minute hangup', () => {
    expect(DEFAULT_IDLE_HANGUP_SECONDS).toBe(300)
  })
})
