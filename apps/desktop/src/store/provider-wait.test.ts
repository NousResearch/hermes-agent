import { describe, expect, it } from 'vitest'

import { parseModelLoadWait, providerWaitText } from './provider-wait'

// The load-notice string is minted by the backend
// (agent/chat_completion_helpers._managed_local_load_notice) and parsed
// here — these tests pin the desktop side of that cross-language contract
// (the backend pins its side in tests/hermes_cli/test_load_progress.py).
describe('providerWaitText', () => {
  it('accepts the managed-local load frame', () => {
    const frame = '⏳ loading Qwen3.6-35B-A3B-UD-Q4_K_M into memory — 42% (responses start once the model is loaded)'

    expect(providerWaitText(frame)).toBe(frame)
  })

  it('still accepts classic wait frames and rejects spinner noise', () => {
    expect(providerWaitText('⏳ waiting on local-model — 30s with no output yet')).not.toBe('')
    expect(providerWaitText('◉_◉ cogitating...')).toBe('')
  })

  it('accepts the near-deadline update minted by agent/chat_completion_wait_notice.wait_notice_text', () => {
    // Exact output of wait_notice_text('gpt-5.5', 290, 'first_event', ('TTFB', 10)); rejecting it
    // left the status row on the stale first notice until the reconnect.
    const frame =
      '⏳ still waiting on gpt-5.5 — 290s waiting for the first provider event (auto-reconnect: TTFB watchdog in 10s)'

    expect(providerWaitText(frame)).toBe(frame)
  })

  it('accepts the reset-window retry notice minted by agent/turn_recovery.compute_error_backoff', () => {
    // Exact shape of turn_recovery.py's `_live_reason` when a reset hint is available
    // (_wait_reason is exactly "Rate limited" or "Provider overloaded", lowercased here) —
    // rejecting it left Desktop's status row on stale/empty text while CLI/TUI (which render
    // the raw diagnostic text with no regex gate) already showed the reset window.
    const rateLimited = '⏳ rate limited — resets in ~13m, retrying in 30s (attempt 2/5)'
    const overloaded = '⏳ provider overloaded — resets in ~5m, retrying in 60s (attempt 1/5)'

    expect(providerWaitText(rateLimited)).toBe(rateLimited)
    expect(providerWaitText(overloaded)).toBe(overloaded)
  })
})

describe('parseModelLoadWait', () => {
  it('extracts model and percent from a load frame', () => {
    expect(
      parseModelLoadWait(
        '⏳ loading Qwen3.6-35B-A3B-UD-Q4_K_M into memory — 42% (responses start once the model is loaded)'
      )
    ).toEqual({ kind: 'load', model: 'Qwen3.6-35B-A3B-UD-Q4_K_M', percent: 42 })
  })

  it('extracts the percent from a prefill frame', () => {
    expect(parseModelLoadWait('⚙ processing prompt — 31%')).toEqual({
      kind: 'prefill',
      model: '',
      percent: 31
    })
  })

  it('parses a percentless prefill frame with a null percent (no fake bar)', () => {
    expect(parseModelLoadWait('⚙ processing prompt')).toEqual({
      kind: 'prefill',
      model: '',
      percent: null
    })
  })

  it('returns null for every other wait frame', () => {
    expect(parseModelLoadWait('⏳ waiting on qwen — 30s with no output yet')).toBeNull()
    expect(parseModelLoadWait('⚠ no output from provider for 900s — reconnecting...')).toBeNull()
    expect(parseModelLoadWait('')).toBeNull()
  })

  it('clamps out-of-range percents', () => {
    expect(parseModelLoadWait('⏳ loading m into memory — 999%')?.percent).toBe(100)
  })
})

describe('providerWaitText accepts prefill frames', () => {
  it('passes the ⚙ processing-prompt frame through', () => {
    const frame = '⚙ processing prompt — 31%'

    expect(providerWaitText(frame)).toBe(frame)
  })
})
