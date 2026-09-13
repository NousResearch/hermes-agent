import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { renderMessageStream } from './test-harness'

afterEach(cleanup)

it('rejects stale snapshots and terminal frames across generations and owner epochs', () => {
  const stream = renderMessageStream('authority-session')

  const send = (type: string, authority_epoch: number | undefined, execution_generation: number | undefined, running?: boolean) =>
    act(() => stream.handleEvent({ type, session_id: 'authority-session', authority_epoch, execution_generation, payload: { running } }))

  send('message.start', 1, 9)
  send('session.info', 1, 8, false)
  expect(stream.state().busy).toBe(true)
  send('message.complete', undefined, undefined)
  expect(stream.state().busy).toBe(true)
  send('session.info', 2, 1, true)
  send('message.complete', 1, 10)
  expect(stream.state().busy).toBe(true)
  send('session.info', 2, 1, false)
  expect(stream.state().busy).toBe(false)
  send('session.info', 2, 1, true)
  expect(stream.state().busy).toBe(false)
})

it('accepts a versioned terminal snapshot before optimistic pre-start grace expires', () => {
  const stream = renderMessageStream('early-terminal')
  stream.states.set('early-terminal', { ...stream.state(), busy: true, awaitingResponse: true, turnLive: false, turnStartedAt: Date.now() })
  act(() => stream.handleEvent({ type: 'session.info', session_id: 'early-terminal', authority_epoch: 1, execution_generation: 1, payload: { execution_state: 'error', running: false } }))
  expect(stream.state().busy).toBe(false)
  expect(stream.state().awaitingResponse).toBe(false)
})
