import type { GatewayEvent } from '@hermes/shared'
// Every turn ends with two frames: message.complete, then the turn worker's
// end-of-turn session.info (running=false), which core sends after it has
// released the session. A turn the desktop starts right after a completion (a
// queued message drained as soon as the reply completes, a send right after
// Stop) can have its message.start delivered BEFORE that report. The report is
// about the turn that just completed: settled on the new turn, it dropped Stop,
// cleared the turn clock and un-pended the reply while the agent was still
// working, and every later write of the cached state put that settle back.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { finalizeStoppedMessages } from '@/app/session/hooks/use-prompt-actions/rewind'
import type { ClientSessionState } from '@/app/types'
import { finalizeInterruptedMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const CHAT = 'late-turn-end-chat'
const RUNTIME = 'late-turn-end-runtime'

let stream: MessageStreamHarness

const emit = (event: Omit<GatewayEvent, 'session_id'>) => act(() => stream.handleEvent({ ...event, session_id: RUNTIME }))

const turnEndReport = () => emit({ payload: { running: false, stored_session_id: CHAT }, type: 'session.info' })

const flushDeltas = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
  })

// What the composer's send does to the session before the gateway answers
// (use-prompt-actions/submit.ts seedOptimistic): the user's row, busy, the clock.
const submit = (text: string) =>
  act(() => {
    const state = stream.state(RUNTIME)

    stream.states.set(RUNTIME, {
      ...state,
      messages: [
        ...finalizeInterruptedMessages(state.messages, state.streamId),
        { id: `user-${text}`, role: 'user', parts: [{ type: 'text', text }] }
      ],
      busy: true,
      awaitingResponse: true,
      pendingBranchGroup: null,
      sawAssistantPayload: false,
      streamId: null,
      interrupted: false,
      turnStartedAt: Date.now()
    })
  })

// What Stop does on screen before the backend's interrupted completion arrives (cancelRun).
const stop = () =>
  act(() => {
    const state = stream.state(RUNTIME)

    stream.states.set(RUNTIME, {
      ...state,
      messages: finalizeStoppedMessages(state.messages, state.streamId),
      awaitingResponse: false,
      busy: false,
      interrupted: true,
      streamId: null,
      turnLive: false,
      turnStartedAt: null
    })
  })

const reply = () => stream.state(RUNTIME).messages.filter(message => message.role === 'assistant').at(-1)

function mount() {
  vi.useFakeTimers()
  stream = renderMessageStream(RUNTIME, {
    states: new Map<string, ClientSessionState>([[RUNTIME, { ...createClientSessionState(), storedSessionId: CHAT }]])
  })
}

async function work(text: string) {
  emit({ payload: { args: { path: 'NOTES.md' }, name: 'read_file', tool_id: `call-${text}` }, type: 'tool.start' })
  emit({ payload: { name: 'read_file', result: 'ok', tool_id: `call-${text}` }, type: 'tool.complete' })
  emit({ payload: { text }, type: 'message.delta' })
  await flushDeltas()
}

async function completedTurn(text: string) {
  submit(text)
  emit({ payload: {}, type: 'message.start' })
  await work(`${text}: answer.`)
  emit({ payload: { status: 'complete', text: `${text}: answer.` }, type: 'message.complete' })
}

function expectRunning() {
  const state = stream.state(RUNTIME)

  expect(state.busy).toBe(true)
  expect(state.turnLive).toBe(true)
  expect(state.turnStartedAt).not.toBeNull()
}

function expectSettled() {
  const state = stream.state(RUNTIME)

  expect(state.busy).toBe(false)
  expect(state.turnLive).toBe(false)
  expect(reply()?.pending).not.toBe(true)
}

describe("a turn that starts before the previous turn's end-of-turn report arrives", () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('keeps a message drained right after a reply running until its own end', async () => {
    mount()
    await completedTurn('first')

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    turnEndReport()
    expectRunning()

    emit({ payload: { text: 'Second' }, type: 'message.delta' })
    await flushDeltas()
    expectRunning()
    expect(reply()?.pending).toBe(true)

    await work('Second: answer.')
    expectRunning()

    emit({ payload: { status: 'complete', text: 'Second: answer.' }, type: 'message.complete' })
    turnEndReport()
    expectSettled()
  })

  it('keeps the message held for a paused reply running after the continuation completes', async () => {
    mount()
    // A host plugin pauses the reply to apply a grant (session.interrupt `resume`), then continues it with a
    // hidden prompt; the send held meanwhile goes the moment the continuation completes.
    submit('read the doc')
    emit({ payload: {}, type: 'message.start' })
    emit({ payload: { args: { path: 'DOC.md' }, name: 'read_file', tool_id: 'call-paused' }, type: 'tool.start' })
    emit({ payload: { paused: true, status: 'interrupted', text: '' }, type: 'message.complete' })
    turnEndReport()
    emit({ payload: {}, type: 'message.start' })
    await work('The doc says hello.')
    emit({ payload: { status: 'complete', text: 'The doc says hello.' }, type: 'message.complete' })

    submit('read the doc again')
    emit({ payload: {}, type: 'message.start' })
    turnEndReport()
    expectRunning()

    await work('Same file, read again.')
    expectRunning()
  })

  it('keeps a message sent right after Stop running', async () => {
    mount()
    submit('first')
    emit({ payload: {}, type: 'message.start' })
    await work('First, cut')
    stop()
    emit({ payload: { status: 'interrupted', text: 'First, cut' }, type: 'message.complete' })

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    turnEndReport()
    expectRunning()
  })

  it("still settles on the live turn's own report when its completion never arrived", async () => {
    mount()
    await completedTurn('first')
    turnEndReport()

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    await work('Second, cut off')
    // message.complete lost (a reconnect gap): the end-of-turn report is the only settle signal.
    turnEndReport()

    expectSettled()
  })

  it("takes the late report as the completed turn's once: the next turn still settles on its own", async () => {
    mount()
    await completedTurn('first')

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    turnEndReport()
    expectRunning()

    await work('Second, cut off')
    turnEndReport()

    expectSettled()
  })

  it('owes nothing for a turn its report already settled ahead of the completion', async () => {
    mount()
    submit('first')
    emit({ payload: {}, type: 'message.start' })
    await work('First: answer.')
    // The completion is reordered behind the report (#119569).
    turnEndReport()
    emit({ payload: { status: 'complete', text: 'First: answer.' }, type: 'message.complete' })

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    await work('Second, cut off')
    turnEndReport()

    expectSettled()
  })
})
