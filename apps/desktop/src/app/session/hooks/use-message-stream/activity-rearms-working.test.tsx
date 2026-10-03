// #106451: the Desktop UI dropped to an idle state while the turn was still
// producing activity rows (tool/text events kept arriving after a premature
// settle), so it looked finished while still executing. Stream activity must
// re-assert working; only a terminal event may clear it.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'activity-rearm-session'

const sessionStates = new Map<string, ClientSessionState>()
let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID, { states: sessionStates })
}

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}) {
  act(() => stream.handleEvent({ payload, session_id: SID, type }))
}

describe('activity after a premature settle re-asserts working (#106451)', () => {
  afterEach(() => {
    cleanup()
    sessionStates.clear()
    vi.restoreAllMocks()
  })

  it('a tool.start after message.complete re-arms busy', () => {
    mountStream()
    emit('message.start')
    expect(stream.state().busy).toBe(true)

    emit('tool.start', { args: { command: 'true' }, name: 'terminal', tool_id: 't1' })
    emit('message.complete', { text: 'done' })
    expect(stream.state().busy).toBe(false)

    // The turn is demonstrably still executing: a new tool starts.
    emit('tool.start', { args: { command: 'true' }, name: 'terminal', tool_id: 't2' })
    expect(stream.state().busy).toBe(true)
  })

  it('a message.delta after message.complete re-arms busy', () => {
    mountStream()
    emit('message.start')
    emit('message.complete', { text: 'done' })
    expect(stream.state().busy).toBe(false)

    emit('message.delta', { text: 'still going' })
    expect(stream.state().busy).toBe(true)
  })

  it('stays idle when the user stopped the turn', () => {
    sessionStates.set(SID, { ...createClientSessionState(), busy: false, interrupted: true })
    mountStream()

    emit('tool.start', { args: { command: 'true' }, name: 'terminal', tool_id: 't1' })
    emit('message.delta', { text: 'late frame' })

    expect(stream.state().busy).toBe(false)
  })
})
