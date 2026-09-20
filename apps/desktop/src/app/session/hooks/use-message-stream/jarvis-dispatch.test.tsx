import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $jarvisUi, resetJarvisSession } from '@/app/jarvis/store'
import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'jarvis-dispatch-session'

const sessionStates = new Map<string, ClientSessionState>()
let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID, { states: sessionStates })
}

function emit(event: RpcEvent) {
  act(() => stream.handleEvent(event))
}

describe('Jarvis projection from the production message stream', () => {
  beforeEach(() => {
    sessionStates.clear()
    resetJarvisSession(SID)
  })

  afterEach(() => {
    cleanup()
    sessionStates.clear()
    resetJarvisSession(SID)
    vi.restoreAllMocks()
  })

  it('projects an error message.complete as task.failed after the message handler accepts it', () => {
    mountStream()

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({
      payload: {
        error: 'backend said no',
        rendered: 'rendered fallback',
        status: 'error',
        task_id: 'task-1',
        text: 'text fallback',
      },
      session_id: SID,
      type: 'message.complete',
    })

    expect($jarvisUi.get().task).toEqual({ id: 'task-1', phase: 'failed' })
    expect($jarvisUi.get().activity.map(event => event.type)).toEqual(['task.running', 'task.failed'])
    expect($jarvisUi.get().activity.at(-1)?.detail).toBe('backend said no')
  })

  it('does not project a stale message.start that the message handler leaves interrupted', () => {
    sessionStates.set(SID, { ...createClientSessionState(), interrupted: true })
    mountStream()

    emit({ payload: { task_id: 'task-1' }, session_id: SID, type: 'message.start' })

    expect(stream.state().interrupted).toBe(true)
    expect($jarvisUi.get().task).toEqual({ id: null, phase: 'idle' })
    expect($jarvisUi.get().activity).toEqual([])
  })

  it('projects the explicitly supported Jarvis voice family without a production handler', () => {
    mountStream()

    emit({ payload: { state: 'recording' }, session_id: SID, type: 'voice.status' } as RpcEvent)

    expect($jarvisUi.get().voice).toBe('listening')
    expect($jarvisUi.get().activity.map(event => event.type)).toEqual(['voice.listening'])

    emit({ payload: { text: 'hello' }, session_id: SID, type: 'voice.transcript' } as RpcEvent)

    expect($jarvisUi.get().voice).toBe('idle')
    expect($jarvisUi.get().activity.map(event => event.type)).toEqual(['voice.listening', 'voice.stopped'])

    emit({ payload: {}, session_id: SID, type: 'voice.status' } as RpcEvent)
    emit({ payload: {}, session_id: SID, type: 'voice.interrupted' } as RpcEvent)

    expect($jarvisUi.get().voice).toBe('idle')
    expect($jarvisUi.get().activity.map(event => event.type)).toEqual([
      'voice.listening',
      'voice.stopped',
      'voice.stopped',
      'voice.stopped',
    ])
  })

  it('does not project events that no production handler consumes and Jarvis does not explicitly support', () => {
    mountStream()

    emit({ payload: { state: 'recording' }, session_id: SID, type: 'voice.nope' } as RpcEvent)

    expect($jarvisUi.get().voice).toBe('idle')
    expect($jarvisUi.get().activity).toEqual([])
  })
})
