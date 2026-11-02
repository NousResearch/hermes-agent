import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { cacheHitLabel, contextBarLabel, tokensPerSecondLabel, usageContextLabel } from '@/lib/statusbar'
import { $currentUsage } from '@/store/session'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'

// What gateway/session_usage.py::publish_usage sends a local viewer: the embedded ticker's nested
// `usage` view plus the deprecated top-level counters (#140194). A hand-written copy: this file pins
// how Desktop reads that shape, not that the gateway produces it (tests/gateway/test_local_viewer_usage.py
// pins the producer).
const GATEWAY_FRAME = {
  active_subagents: 0,
  api_calls: 3,
  avg_latency_s: 3.0,
  cache_read_tokens: 36_000,
  cache_write_tokens: 800,
  completion_tokens: 380,
  compression_count: 0,
  config_context_length: 200_000,
  context_length: 200_000,
  context_source: 'provider_usage',
  context_tokens: 42_000,
  fallback_model: null,
  input_tokens: 1_200,
  last_prompt_tokens: 42_000,
  last_turn_tokens: 0,
  model: 'claude-opus-5-5',
  output_tokens: 380,
  prompt_tokens: 42_000,
  provider: 'anthropic',
  rate_limit: null,
  reasoning_tokens: 0,
  tokens_per_s: 63.33,
  total_tokens: 42_380,
  usage: {
    active_subagents: 0,
    avg_latency_s: 3.0,
    avg_tps: 63.3,
    cache_hit_pct: 86,
    calls: 3,
    completion: 380,
    compressions: 0,
    context_estimated: false,
    context_max: 200_000,
    context_percent: 21,
    context_source: 'provider_usage',
    context_used: 42_000,
    input: 1_200,
    model: 'claude-opus-5-5',
    output: 380,
    prompt: 42_000,
    reasoning: 0,
    total: 42_380
  }
}

const METERS = ['[██░░░░░░░░] 21%', '42k/200k', '86%', '63 t/s']

const meters = (usage: Parameters<typeof contextBarLabel>[0]) => [
  contextBarLabel(usage),
  usageContextLabel(usage),
  cacheHitLabel(usage),
  tokensPerSecondLabel(usage)
]

// $currentUsage mirrors the primary session; ClientSessionState.usage drives
// the same status bar when a secondary tile is focused.
const BASELINE = { calls: 2, compressions: 1, input: 500, output: 40, total: 540 }

let stream: MessageStreamHarness
let sessionStates = new Map<string, ClientSessionState>()

function mountStream() {
  stream = renderMessageStream(SID, { states: sessionStates })
}

describe('useMessageStream status-bar usage scoping', () => {
  beforeEach(() => {
    sessionStates = new Map([[SID, { ...createClientSessionState(), usage: { ...BASELINE } }]])
    $currentUsage.set({ ...BASELINE })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('merges a live session.usage tick from the focused session', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { usage: { compressions: 2, context_percent: 42, input: 1200, total: 1280 } },
        session_id: SID,
        type: 'session.usage'
      })
    )

    // Merge, not replace: fields absent from the tick keep their prior values.
    expect($currentUsage.get()).toEqual({ ...BASELINE, compressions: 2, context_percent: 42, input: 1200, total: 1280 })
    expect(sessionStates.get(SID)?.usage).toEqual({
      ...BASELINE,
      compressions: 2,
      context_percent: 42,
      input: 1200,
      total: 1280
    })
  })

  it('normalizes a canonical top-level session.usage tick for the status bar', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: {
          api_calls: 3,
          cache_read_tokens: 800,
          input_tokens: 1_200,
          output_tokens: 80,
          prompt_tokens: 1_000,
          tokens_per_s: 41.64,
          total_tokens: 1_280
        },
        session_id: SID,
        type: 'session.usage'
      })
    )

    expect($currentUsage.get()).toEqual({
      ...BASELINE,
      avg_tps: 41.6,
      cache_hit_pct: 80,
      calls: 3,
      input: 1_200,
      output: 80,
      prompt: 1_000,
      total: 1_280
    })
    expect(sessionStates.get(SID)?.usage).toEqual($currentUsage.get())
  })

  it('paints the status-bar meters from either half of the gateway frame shape', () => {
    $currentUsage.set({ calls: 0, input: 0, output: 0, total: 0 })
    mountStream()

    act(() => stream.handleEvent({ payload: GATEWAY_FRAME, session_id: SID, type: 'session.usage' }))

    expect(meters($currentUsage.get())).toEqual(METERS)

    // A bundle older than usageFromTick merged only `payload.usage`; the nested view alone fills the
    // same meters, and the deprecated counters alone still map to them too.
    const { usage: nested, ...counters } = GATEWAY_FRAME

    expect(meters(nested)).toEqual(METERS)
    $currentUsage.set({ calls: 0, input: 0, output: 0, total: 0 })
    act(() => stream.handleEvent({ payload: counters, session_id: SID, type: 'session.usage' }))
    expect(meters($currentUsage.get())).toEqual(METERS)
  })

  it('maps latency and active subagents from a canonical tick the same way Ink does', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { active_subagents: 2, api_calls: 1, avg_latency_s: 0.84, total_tokens: 10 },
        session_id: SID,
        type: 'session.usage'
      })
    )

    expect($currentUsage.get()).toMatchObject({ active_subagents: 2, avg_latency_s: 0.8, calls: 1, total: 10 })
  })

  it('keeps the compression count of an unsized compressor, as session_usage.tui_usage does', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { api_calls: 1, compression_count: 2, context_length: null, context_source: 'local_estimate' },
        session_id: SID,
        type: 'session.usage'
      })
    )

    expect($currentUsage.get()).toMatchObject({ calls: 1, compressions: 2 })
  })

  it('merges a partial canonical tick without zeroing counters it does not carry', () => {
    mountStream()

    act(() => stream.handleEvent({ payload: { api_calls: 3 }, session_id: SID, type: 'session.usage' }))
    act(() =>
      stream.handleEvent({
        payload: { context_length: 10_000, last_prompt_tokens: 4_200 },
        session_id: SID,
        type: 'session.usage'
      })
    )

    expect($currentUsage.get()).toMatchObject({ ...BASELINE, calls: 3, context_percent: 42, context_used: 4_200 })
  })

  it('falls back to prompt_tokens and completion_tokens when input_tokens and output_tokens are zero', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: {
          api_calls: 3,
          completion_tokens: 80,
          input_tokens: 0,
          output_tokens: 0,
          prompt_tokens: 1_000,
          total_tokens: 1_080
        },
        session_id: SID,
        type: 'session.usage'
      })
    )

    expect($currentUsage.get()).toEqual({
      ...BASELINE,
      calls: 3,
      completion: 80,
      input: 1_000,
      output: 80,
      prompt: 1_000,
      total: 1_080
    })
  })

  it('caches a background session.usage tick without overwriting the primary status bar', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { usage: { compressions: 9, input: 9999, total: 9999 } },
        session_id: 'background-session',
        type: 'session.usage'
      })
    )

    expect($currentUsage.get()).toEqual(BASELINE)
    expect(sessionStates.get('background-session')?.usage).toEqual({
      calls: 0,
      compressions: 9,
      input: 9999,
      output: 0,
      total: 9999
    })
  })

  it('caches authoritative session.info usage for a background tile', () => {
    sessionStates.set('background-session', { ...createClientSessionState(), usage: null })
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { usage: { calls: 3, compressions: 4, input: 700, output: 60, total: 760 } },
        session_id: 'background-session',
        type: 'session.info'
      })
    )

    expect($currentUsage.get()).toEqual(BASELINE)
    expect(sessionStates.get('background-session')?.usage).toEqual({
      calls: 3,
      compressions: 4,
      input: 700,
      output: 60,
      total: 760
    })
  })

  it('clears a stale background compression count when session.info omits it', () => {
    sessionStates.set('background-session', {
      ...createClientSessionState(),
      usage: { ...BASELINE, compressions: 4, context_percent: 55 }
    })
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { usage: { calls: 3, input: 700, output: 60, total: 760 } },
        session_id: 'background-session',
        type: 'session.info'
      })
    )

    const usage = sessionStates.get('background-session')?.usage
    expect($currentUsage.get()).toEqual(BASELINE)
    expect(usage?.compressions).toBeUndefined()
    expect(usage?.context_percent).toBe(55)
    expect(usage?.calls).toBe(3)
  })

  it('applies message.complete usage from the focused session', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { text: 'done', usage: { calls: 3, compressions: 3, input: 1500, output: 90, total: 1590 } },
        session_id: SID,
        type: 'message.complete'
      })
    )

    expect($currentUsage.get()).toEqual({ calls: 3, compressions: 3, input: 1500, output: 90, total: 1590 })
  })

  it('drops the plan gauge once a complete usage payload omits it, but keeps it across partial ticks', () => {
    const gauge = { id: 'day', label: 'Today', resets_at_ms: null, used_bp: 4_200 }
    sessionStates.set(SID, { ...createClientSessionState(), usage: { ...BASELINE, plan_gauge: gauge } })
    $currentUsage.set({ ...BASELINE, plan_gauge: gauge })
    mountStream()

    // A partial tick never carries the gauge; it must not flicker off.
    act(() => stream.handleEvent({ payload: { usage: { input: 900 } }, session_id: SID, type: 'session.usage' }))
    expect($currentUsage.get().plan_gauge).toEqual(gauge)

    // Both usage builders always send `calls` and send `plan_gauge` only on a window plan.
    const { usage: complete } = GATEWAY_FRAME
    act(() =>
      stream.handleEvent({ payload: { text: 'done', usage: complete }, session_id: SID, type: 'message.complete' })
    )
    expect($currentUsage.get().plan_gauge).toBeUndefined()
    expect(sessionStates.get(SID)?.usage?.plan_gauge).toBeUndefined()

    $currentUsage.set({ ...BASELINE, plan_gauge: gauge })
    act(() => stream.handleEvent({ payload: GATEWAY_FRAME, session_id: SID, type: 'session.usage' }))
    expect($currentUsage.get().plan_gauge).toBeUndefined()
  })

  it('ignores message.complete usage from a background session', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { text: 'done', usage: { calls: 9, compressions: 9, input: 9999, output: 999, total: 9999 } },
        session_id: 'background-session',
        type: 'message.complete'
      })
    )

    expect($currentUsage.get()).toEqual(BASELINE)
  })
})
