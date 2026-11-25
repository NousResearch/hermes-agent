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
import { $compactingSessions, setSessionCompacting } from '@/store/compaction'
import { clearAllPrompts, hasBlockingPromptRequest, setApprovalRequest } from '@/store/prompts'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const CHAT = 'late-turn-end-chat'
// Past gateway-event/turn-end-report.ts TURN_END_REPORT_WAIT_MS (30 s). A literal, so this file also loads on a
// tree without that module and its cases fail there one by one.
const REPORT_NO_LONGER_EXPECTED_MS = 60_000
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

  // Two short turns can both complete before the first one's report arrives (core sends each after its own
  // git and workspace probes): each completion owes its own report.
  it('keeps the third turn running while the reports of two completed turns arrive', async () => {
    mount()
    await completedTurn('first')
    await completedTurn('second')

    submit('third')
    emit({ payload: {}, type: 'message.start' })
    turnEndReport()
    expectRunning()
    turnEndReport()
    expectRunning()

    await work('Third, cut off')
    // Both are paid: the next report is this turn's own (its completion was lost).
    turnEndReport()
    expectSettled()
  })

  it("keeps the third turn running when the first of two owed reports came before it started", async () => {
    mount()
    await completedTurn('first')
    await completedTurn('second')
    turnEndReport()

    submit('third')
    emit({ payload: {}, type: 'message.start' })
    turnEndReport()
    expectRunning()
  })

  // A report that never comes (lost in a reconnect), or one that came ahead of a stopped turn's completion
  // (Stop leaves the turn stopped here, so its completion still marks a report owed), leaves a mark nothing
  // pays. It lapses, so a later turn still settles on its own report.
  it("lets a stopped turn's unpaid mark lapse: a later turn settles on its own report", async () => {
    mount()
    submit('first')
    emit({ payload: {}, type: 'message.start' })
    await work('First, cut')
    stop()
    turnEndReport()
    emit({ payload: { status: 'interrupted', text: 'First, cut' }, type: 'message.complete' })

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    await work('Second, cut off')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(REPORT_NO_LONGER_EXPECTED_MS)
    })
    // message.complete lost: the report is this turn's own.
    turnEndReport()

    expectSettled()
  })
})

// The report also ends a compaction phase whose `compacted` edge was lost (a reconnect), and clears prompts a
// lost completion left parked. Both are facts about the turn the report is for: taken from the completed
// turn's late report they would end the phase and close the card of the turn now live.
describe("what the completed turn's late report leaves to the turn now live", () => {
  const compacting = () => Object.hasOwn($compactingSessions.get(), RUNTIME)
  const status = (kind: string) => emit({ payload: { kind, text: '' }, type: 'status.update' })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    setSessionCompacting(RUNTIME, false)
    clearAllPrompts(RUNTIME)
  })

  it('keeps the compaction phase of a turn that compacts as it starts, until its own compacted edge', async () => {
    mount()
    await completedTurn('first')

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    status('compacting')
    expect(compacting()).toBe(true)

    turnEndReport()

    expect(compacting()).toBe(true)
    expectRunning()

    status('compacted')
    expect(compacting()).toBe(false)
    expectRunning()
  })

  it('keeps it through the reports of two completed turns', async () => {
    mount()
    await completedTurn('first')
    await completedTurn('second')

    submit('third')
    emit({ payload: {}, type: 'message.start' })
    status('compacting')
    turnEndReport()
    turnEndReport()

    expect(compacting()).toBe(true)
    expectRunning()
  })

  it("keeps a manual /compress phase too: the completed turn's report is not its `ready`", async () => {
    mount()
    await completedTurn('first')

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    status('compressing')
    turnEndReport()

    expect(compacting()).toBe(true)

    status('ready')
    expect(compacting()).toBe(false)
  })

  it("still ends the phase on the live turn's own report when its compacted edge and completion never arrived", async () => {
    mount()
    await completedTurn('first')
    turnEndReport()

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    status('compacting')
    expect(compacting()).toBe(true)

    // Both lost in a reconnect: the report is this turn's own and the only terminal fact.
    turnEndReport()

    expect(compacting()).toBe(false)
    expectSettled()
  })

  it('still ends the phase of a session no turn is live on (a report after a reconnect)', () => {
    mount()
    act(() => setSessionCompacting(RUNTIME, true))

    turnEndReport()

    expect(compacting()).toBe(false)
  })

  it('keeps the phase through a running heartbeat of the turn that compacts', () => {
    mount()
    submit('first')
    emit({ payload: {}, type: 'message.start' })
    status('compacting')

    emit({ payload: { running: true, stored_session_id: CHAT }, type: 'session.info' })

    expect(compacting()).toBe(true)
    expectRunning()
  })

  it('keeps an approval the turn now live is waiting on', async () => {
    mount()
    await completedTurn('first')

    submit('second')
    emit({ payload: {}, type: 'message.start' })
    act(() => setApprovalRequest({ command: 'rm -rf build', description: 'delete the build folder', sessionId: RUNTIME }))

    turnEndReport()

    expect(hasBlockingPromptRequest(RUNTIME)).toBe(true)
    expectRunning()
  })
})
