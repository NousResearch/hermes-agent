import { describe, expect, it } from 'vitest'

import { parseGenericWait, parseModelLoadWait, providerWaitText } from './provider-wait'

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
})

describe('providerWaitText accepts retry and auto-recovery frames', () => {
  // Minted by agent/turn_recovery.py (retry backoff) and
  // agent/turn_recovery_autorecover.ladder_notice (outage ladder). Rejecting
  // them left a bare timer for minutes while the backend slept between retries.
  it.each([
    '⏳ rate limited — resets in 2m, retrying in 30s (attempt 2/3)',
    '⏳ provider overloaded — resets in 1m, retrying in 15s (attempt 1/3)',
    '⏳ waiting on provider — retrying in 6s (attempt 1/3)',
    '⏳ Provider temporarily unavailable — retrying automatically in 18s (cycle 1/5); press Esc to stop'
  ])('%s', frame => {
    expect(providerWaitText(frame)).toBe(frame)
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
      percent: 31,
      processed: null,
      total: null,
      totalEstimated: false
    })
  })

  it('extracts counts and percent from a prefill frame with a real total', () => {
    expect(parseModelLoadWait('⚙ processing prompt — 46,500 / 80,000 tokens (58%)')).toEqual({
      kind: 'prefill',
      model: '',
      percent: 58,
      processed: 46500,
      total: 80000,
      totalEstimated: false
    })
  })

  it('marks an estimated total with totalEstimated', () => {
    expect(parseModelLoadWait('⚙ processing prompt — 12,288 / ~39,551 tokens (31%)')).toEqual({
      kind: 'prefill',
      model: '',
      percent: 31,
      processed: 12288,
      total: 39551,
      totalEstimated: true
    })
  })

  it('extracts the processed count from a tokens-read prefill frame (no fake percent)', () => {
    expect(parseModelLoadWait('⚙ processing prompt — 12,288 tokens read')).toEqual({
      kind: 'prefill',
      model: '',
      percent: null,
      processed: 12288,
      total: null,
      totalEstimated: false
    })
  })

  it('parses a percentless prefill frame with nulls throughout (no fake bar)', () => {
    expect(parseModelLoadWait('⚙ processing prompt')).toEqual({
      kind: 'prefill',
      model: '',
      percent: null,
      processed: null,
      total: null,
      totalEstimated: false
    })
  })

  it('parses the queue frame', () => {
    expect(parseModelLoadWait('⏳ waiting for a free slot — the local server is busy with another request')).toEqual({
      kind: 'queued',
      model: ''
    })
  })

  it('returns null for every other wait frame', () => {
    expect(parseModelLoadWait('⏳ waiting on qwen — 30s with no output yet')).toBeNull()
    expect(parseModelLoadWait('⚠ no output from provider for 900s — reconnecting...')).toBeNull()
    expect(parseModelLoadWait('')).toBeNull()
  })

  it('clamps out-of-range percents', () => {
    expect(parseModelLoadWait('⏳ loading m into memory — 999%')).toEqual({
      kind: 'load',
      model: 'm',
      percent: 100
    })
  })
})

describe('providerWaitText accepts prefill frames', () => {
  it('passes the ⚙ processing-prompt frame through', () => {
    const frame = '⚙ processing prompt — 31%'

    expect(providerWaitText(frame)).toBe(frame)
  })

  it('passes counts-bearing prefill frames through', () => {
    expect(providerWaitText('⚙ processing prompt — 46,500 / 80,000 tokens (58%)')).not.toBe('')
    expect(providerWaitText('⚙ processing prompt — 12,288 tokens read')).not.toBe('')
  })

  it('passes the queue frame through', () => {
    const frame = '⏳ waiting for a free slot — the local server is busy with another request'

    expect(providerWaitText(frame)).toBe(frame)
  })
})

describe('parseGenericWait', () => {
  it('parses the first-chunk wait without its frozen seconds', () => {
    expect(parseGenericWait('⏳ waiting on qwen — 60s waiting for the first stream chunk')).toEqual({
      kind: 'waiting',
      model: 'qwen',
      phase: 'first-chunk',
      watchdog: null
    })
  })

  it('parses the near-deadline first-event wait and keeps only the watchdog label', () => {
    expect(
      parseGenericWait(
        '⏳ still waiting on gpt-5.5 — 290s waiting for the first provider event (auto-reconnect: TTFB watchdog in 10s)'
      )
    ).toEqual({
      kind: 'waiting',
      model: 'gpt-5.5',
      phase: 'first-event',
      watchdog: 'TTFB watchdog'
    })
  })

  it('parses the post-chunk and pre-progress phases', () => {
    expect(parseGenericWait('⏳ waiting on qwen — 90s stream open; 90s without stream output')).toEqual({
      kind: 'waiting',
      model: 'qwen',
      phase: 'post-chunk',
      watchdog: null
    })
    expect(
      parseGenericWait('⏳ waiting on qwen — 90s provider stream open; 90s without substantive model progress')
    ).toEqual({ kind: 'waiting', model: 'qwen', phase: 'pre-progress', watchdog: null })
    expect(parseGenericWait('⏳ waiting on qwen — 90s provider stream active; 90s without stream events')).toEqual({
      kind: 'waiting',
      model: 'qwen',
      phase: 'post-event',
      watchdog: null
    })
  })

  it('parses the reconnect diagnostic as a model-less wait', () => {
    expect(parseGenericWait('⚠ no output from provider for 900s — reconnecting...')).toEqual({
      kind: 'waiting',
      model: '',
      phase: 'reconnect',
      watchdog: null
    })
  })

  it('returns null for unrecognised frames (they render verbatim as before)', () => {
    expect(parseGenericWait('⏳ waiting on local-model — 30s with no output yet')).toBeNull()
    expect(parseGenericWait('⏳ rate limited — resets in 2m, retrying in 30s (attempt 2/3)')).toBeNull()
    expect(parseGenericWait('')).toBeNull()
  })
})
