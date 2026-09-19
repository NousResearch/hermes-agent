import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $currentUsage } from '@/store/session'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
// $currentUsage mirrors the primary session; ClientSessionState.usage drives
// the same status bar when a secondary tile is focused.
const BASELINE = { calls: 2, input: 500, output: 40, total: 540 }

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
        payload: { usage: { context_percent: 42, input: 1200, total: 1280 } },
        session_id: SID,
        type: 'session.usage'
      })
    )

    // Merge, not replace: fields absent from the tick keep their prior values.
    expect($currentUsage.get()).toEqual({ ...BASELINE, context_percent: 42, input: 1200, total: 1280 })
    expect(sessionStates.get(SID)?.usage).toEqual({
      ...BASELINE,
      context_percent: 42,
      input: 1200,
      total: 1280
    })
  })

  it('caches a background session.usage tick without overwriting the primary status bar', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { usage: { input: 9999, total: 9999 } },
        session_id: 'background-session',
        type: 'session.usage'
      })
    )

    expect($currentUsage.get()).toEqual(BASELINE)
    expect(sessionStates.get('background-session')?.usage).toEqual({
      calls: 0,
      input: 9999,
      output: 0,
      total: 9999
    })
  })

  it('applies message.complete usage from the focused session', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { text: 'done', usage: { calls: 3, input: 1500, output: 90, total: 1590 } },
        session_id: SID,
        type: 'message.complete'
      })
    )

    expect($currentUsage.get()).toEqual({ calls: 3, input: 1500, output: 90, total: 1590 })
  })

  it('ignores message.complete usage from a background session', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { text: 'done', usage: { calls: 9, input: 9999, output: 999, total: 9999 } },
        session_id: 'background-session',
        type: 'message.complete'
      })
    )

    expect($currentUsage.get()).toEqual(BASELINE)
  })
})

describe('useMessageStream per-turn message usage', () => {
  beforeEach(() => {
    sessionStates = new Map([[SID, { ...createClientSessionState() }]])
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  function completedAssistantMessage() {
    return stream.state(SID).messages.find(m => m.role === 'assistant' && m.completedAt !== undefined)
  }

  it('forwards exact provider usage onto the completed assistant message', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { text: 'The answer is 42.', usage: { input: 12, output: 7, total: 19 } },
        session_id: SID,
        type: 'message.complete'
      })
    )

    const msg = completedAssistantMessage()
    expect(msg).toBeDefined()
    expect(msg?.usage).toEqual({ input: 12, output: 7, total: 19 })
  })

  it('records the per-turn delta when a prior session baseline exists', () => {
    // Seed a running session-usage baseline, then open the turn with
    // message.start (which snapshots the baseline). message.complete reports
    // the CUMULATIVE total, so the chip receives the per-turn delta.
    sessionStates = new Map([
      [SID, { ...createClientSessionState(), usage: { calls: 1, input: 100, output: 50, total: 150 } }]
    ])
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { text: 'Thinking…' },
        session_id: SID,
        type: 'message.start'
      })
    )

    act(() =>
      stream.handleEvent({
        payload: { text: 'One more turn.', usage: { calls: 2, input: 140, output: 80, total: 220 } },
        session_id: SID,
        type: 'message.complete'
      })
    )

    const msg = completedAssistantMessage()
    expect(msg?.usage).toEqual({ input: 40, output: 30, total: 70 })
  })

  it('omits usage when the gateway reports none for the turn', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { text: 'No usage reported.' },
        session_id: SID,
        type: 'message.complete'
      })
    )

    const msg = completedAssistantMessage()
    expect(msg?.usage).toBeUndefined()
  })
})
