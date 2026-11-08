import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId } from '@/store/session'
import {
  $sessionStates,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState,
  setLiveTurnBackend
} from '@/store/session-states'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const SID = 'child-watch'
const cleanups: (() => void)[] = []

const failed = () => ({
  ...createClientSessionState(),
  heartbeatSettledStreamId: 'watched-turn',
  messages: [
    {
      id: 'watched-turn',
      role: 'assistant' as const,
      parts: [],
      pending: false,
      error: 'Hermes ended this turn without a reply.',
      errorSurface: { code: 'no_reply', layer: 'runtime' as const, retryable: true }
    }
  ]
})

beforeEach(() => vi.useFakeTimers())
afterEach(() => {
  cleanups.splice(0).forEach(fn => fn())
  cleanup()
  clearAllSessionStates()
  $activeSessionId.set(null)
  vi.clearAllTimers()
  vi.useRealTimers()
})

const flushDeltas = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS * 10)
  })

const send = (stream: MessageStreamHarness, type: GatewayEventName, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ type, session_id: SID, payload } as Parameters<MessageStreamHarness['handleEvent']>[0]))

// The production watchdog settles an empty placeholder and stamps a real
// no-payload notice on a mounted stream. `refreshTranscript` lets a test hold
// the history recovery that runs between the settle and the stamp.
async function settledTurn(refreshTranscript?: () => Promise<void>) {
  $activeSessionId.set(SID)
  const states = new Map<string, ClientSessionState>()

  const unsubscribe = $sessionStates.subscribe(snapshot => {
    if (snapshot[SID]) {
      states.set(SID, snapshot[SID])
    }
  })

  const release = setLiveTurnBackend({
    request: (async () => ({ sessions: [{ id: SID, status: 'idle' }] })) as never,
    ...(refreshTranscript ? { refreshTranscript } : {})
  })

  cleanups.push(release, unsubscribe)
  publishSessionState(SID, {
    ...createClientSessionState(),
    storedSessionId: 'child-stored',
    busy: true,
    awaitingResponse: true,
    turnLive: true,
    streamId: 'empty-placeholder',
    messages: [{ id: 'empty-placeholder', role: 'assistant', parts: [], pending: true }]
  })

  const stream = renderMessageStream(SID, {
    states,
    updateSessionState: (id, updater) => {
      const next = updater($sessionStates.get()[id])
      publishSessionState(id, next)

      return next
    }
  })

  noteSessionEvent(SID)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(LIVE_TURN_EVENT_SILENCE_MS)
  })

  return stream
}

async function settledNotice() {
  const stream = await settledTurn()
  const notice = stream.state().messages.at(-1)
  expect(notice?.errorSurface?.code).toBe('no_reply')
  expect(notice?.id).not.toBe('empty-placeholder')

  return stream
}

it('late child text clears only the retry card of its heartbeat-settled turn', async () => {
  const stream = renderMessageStream(SID, { states: new Map([[SID, failed()]]) })
  act(() =>
    stream.handleEvent({
      type: 'message.delta',
      session_id: SID,
      payload: { text: 'Python call hung; retrying with curl.' }
    })
  )
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })
  expect(stream.state().messages).toHaveLength(1)
  expect(chatMessageText(stream.state().messages[0])).toContain('retrying with curl')
  expect(stream.state().messages[0].error).toBeUndefined()
  expect(stream.state().messages[0].errorSurface).toBeUndefined()
})

it('late final reply clears the same card', () => {
  const stream = renderMessageStream(SID, { states: new Map([[SID, failed()]]) })
  act(() => stream.handleEvent({ type: 'message.complete', session_id: SID, payload: { text: 'Finished.' } }))
  expect(stream.state().messages).toHaveLength(1)
  expect(stream.state().messages[0].errorSurface).toBeUndefined()
  expect(chatMessageText(stream.state().messages[0])).toBe('Finished.')
})

it('a newer turn keeps the earlier no-reply failure', async () => {
  const stream = renderMessageStream(SID, { states: new Map([[SID, failed()]]) })
  act(() => stream.handleEvent({ type: 'message.start', session_id: SID, payload: {} }))
  act(() => stream.handleEvent({ type: 'message.delta', session_id: SID, payload: { text: 'A new answer.' } }))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })
  expect(stream.state().messages[0].errorSurface?.code).toBe('no_reply')
  expect(chatMessageText(stream.state().messages[1])).toBe('A new answer.')
})

it('a real terminal error replaces the inferred no-reply card', () => {
  const stream = renderMessageStream(SID, { states: new Map([[SID, failed()]]) })
  act(() =>
    stream.handleEvent({
      type: 'message.complete',
      session_id: SID,
      payload: {
        status: 'error',
        text: 'Error: provider timeout',
        error: 'provider timeout',
        error_surface: { code: 'provider_timeout', layer: 'provider', retryable: true }
      }
    })
  )
  expect(stream.state().messages[0].error).toBe('provider timeout')
  expect(stream.state().messages[0].errorSurface?.code).toBe('provider_timeout')
})

it.each(['message.delta', 'message.complete'] as const)('real no-payload watchdog notice is replaced by late %s', async type => {
  const stream = await settledNotice()
  send(stream, type, { text: 'The child continued and answered.' })
  await flushDeltas()
  expect(stream.state().messages).toHaveLength(1)
  expect(chatMessageText(stream.state().messages[0])).toBe('The child continued and answered.')
  expect(stream.state().messages[0].errorSurface).toBeUndefined()
  expect(stream.state().messages[0].error).toBeUndefined()
})

it.each(['message.delta', 'message.complete'] as const)(
  'a bare error on the notice survives a late %s and drops the inferred descriptor',
  async type => {
    const stream = await settledNotice()
    send(stream, 'error', { message: 'Provider transport disconnected' })
    send(stream, type, { text: 'The late answer arrived.' })
    await flushDeltas()
    const failure = stream.state().messages.find(message => message.error === 'Provider transport disconnected')
    expect(failure).toBeDefined()
    expect(failure?.errorSurface?.code).not.toBe('no_reply')
  }
)

it('a late failed completion without a descriptor does not keep the inferred one', async () => {
  const stream = await settledNotice()
  send(stream, 'message.complete', { status: 'error', text: 'Error: upstream reset', error: 'upstream reset' })
  const [failure] = stream.state().messages
  expect(failure.error).toBe('upstream reset')
  expect(failure.errorSurface?.code).not.toBe('no_reply')
})

it.each([
  ['reasoning.delta', { text: 'Checking the result.' }],
  ['message.delta', { text: '  \n' }],
  ['reasoning.available', { text: 'Thought it through.' }],
  ['tool.start', { name: 'terminal', tool_id: 'late-tool', args: { command: 'pwd' } }]
] as const)('%s flushed before the answer keeps the late reply on the notice', async (type, payload) => {
  const stream = await settledNotice()
  send(stream, type, payload)
  await flushDeltas()
  expect(stream.state().messages.at(-1)?.errorSurface?.code).toBe('no_reply')
  send(stream, 'message.delta', { text: 'The late answer arrived.' })
  await flushDeltas()
  send(stream, 'message.complete', { text: 'The late answer arrived.' })
  const replies = stream.state().messages.filter(message => message.role === 'assistant')
  expect(replies).toHaveLength(1)
  expect(replies[0].errorSurface).toBeUndefined()
  expect(chatMessageText(replies[0]).trim()).toBe('The late answer arrived.')
})

it('an empty completion while history recovery is pending still gets the notice', async () => {
  let finishRefresh = () => {}
  const stream = await settledTurn(() => new Promise<void>(resolve => (finishRefresh = resolve)))
  send(stream, 'message.complete', { text: '' })
  await act(async () => {
    finishRefresh()
    await vi.advanceTimersByTimeAsync(0)
  })
  expect(stream.state().messages.at(-1)?.errorSurface?.code).toBe('no_reply')
})

it('late progress then an empty completion while history recovery is pending still gets the notice', async () => {
  let finishRefresh = () => {}
  const stream = await settledTurn(() => new Promise<void>(resolve => (finishRefresh = resolve)))
  send(stream, 'tool.start', { name: 'terminal', tool_id: 'late-tool', args: { command: 'ls' } })
  await flushDeltas()
  send(stream, 'message.complete', { text: '' })
  await act(async () => {
    finishRefresh()
    await vi.advanceTimersByTimeAsync(0)
  })
  expect(stream.state().messages.some(message => message.errorSurface?.code === 'no_reply')).toBe(true)
})

it('a turn armed by a running=true heartbeat writes its own bubble, not the settled one', async () => {
  const stream = renderMessageStream(SID)
  send(stream, 'message.start')
  send(stream, 'message.delta', { text: 'Old reply.' })
  await flushDeltas()
  send(stream, 'session.info', { running: false })
  const settled = stream.state().heartbeatSettledStreamId
  expect(settled).toBeTruthy()
  send(stream, 'session.info', { running: true })
  send(stream, 'message.delta', { text: 'New turn text.' })
  await flushDeltas()
  const old = stream.state().messages.find(message => message.id === settled)
  expect(chatMessageText(old!)).toBe('Old reply.')
  expect(stream.state().messages.map(chatMessageText)).toContain('New turn text.')
})

it('an empty completion of a heartbeat-adopted newer turn does not hand later output to the old bubble', async () => {
  const stream = renderMessageStream(SID)
  send(stream, 'message.start')
  send(stream, 'message.delta', { text: 'Old reply.' })
  await flushDeltas()
  send(stream, 'session.info', { running: false })
  const settled = stream.state().heartbeatSettledStreamId
  send(stream, 'session.info', { running: true })
  send(stream, 'message.delta', { text: 'Newer turn.' })
  await flushDeltas()
  send(stream, 'message.complete', { text: '' })
  send(stream, 'message.delta', { text: ' Later.' })
  await flushDeltas()
  const old = stream.state().messages.find(message => message.id === settled)
  expect(chatMessageText(old!)).toBe('Old reply.')
})

it('an empty late completion of a kept reply does not let an adopted turn overwrite it', async () => {
  const stream = renderMessageStream(SID)
  send(stream, 'message.start')
  send(stream, 'message.delta', { text: 'Old reply.' })
  await flushDeltas()
  send(stream, 'session.info', { running: false })
  send(stream, 'message.complete', { text: '' })
  send(stream, 'session.info', { running: true })
  send(stream, 'message.complete', { text: 'New turn answer.' })
  const texts = stream.state().messages.map(chatMessageText)
  expect(texts).toContain('Old reply.')
  expect(texts).toContain('New turn answer.')
})

it('an adopted turn does not settle its answer onto the previous tool-only bubble', async () => {
  const stream = renderMessageStream(SID)
  send(stream, 'message.start')
  send(stream, 'tool.start', { name: 'terminal', tool_id: 't1', args: { command: 'ls' } })
  send(stream, 'tool.complete', { name: 'terminal', tool_id: 't1', result: 'ok' })
  send(stream, 'session.info', { running: false })
  send(stream, 'message.complete', { text: '' })
  send(stream, 'session.info', { running: true })
  send(stream, 'message.complete', { text: 'New turn answer.' })
  const [tools, reply] = stream.state().messages
  expect(chatMessageText(tools)).toBe('')
  expect(chatMessageText(reply)).toBe('New turn answer.')
})

it('output after a new prompt is submitted does not reclaim the earlier card', async () => {
  const stream = await settledNotice()
  act(() => {
    const current = $sessionStates.get()[SID]
    publishSessionState(SID, {
      ...current,
      awaitingResponse: true,
      busy: true,
      messages: [...current.messages, { id: 'u-next', role: 'user', parts: [{ type: 'text', text: 'next' }] }]
    })
  })
  send(stream, 'message.delta', { text: 'Answer to the next prompt.' })
  await flushDeltas()
  const [card, , reply] = stream.state().messages
  expect(card.errorSurface?.code).toBe('no_reply')
  expect(chatMessageText(card)).toBe('')
  expect(chatMessageText(reply)).toBe('Answer to the next prompt.')
})
