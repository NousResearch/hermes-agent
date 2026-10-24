import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $computerUseBySession,
  clearAllComputerUseStates,
  clearComputerUseState,
  COMPLETED_DISMISS_DELAY_MS
} from '@/store/computer-use'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
const OTHER_SID = 'session-2'

const sessionStates = new Map<string, ClientSessionState>()
let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID, { states: sessionStates })
}

function emit(type: GatewayEvent['type'], payload: any = {}, sessionId = SID) {
  act(() => stream.handleEvent({ payload, session_id: sessionId, type }))
}

/** What Stop does to this store: mark the turn interrupted and retire the pill. */
function stop() {
  sessionStates.set(SID, { ...(sessionStates.get(SID) ?? createClientSessionState()), interrupted: true })
  clearComputerUseState(SID)
}

const cu = () => $computerUseBySession.get()[SID]

describe('computer_use gateway event lifecycle', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    sessionStates.clear()
    clearAllComputerUseStates()
  })

  afterEach(() => {
    cleanup()
    sessionStates.clear()
    clearAllComputerUseStates()
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it('enters drafting state when tool.generating arrives for computer_use', () => {
    mountStream()

    emit('tool.generating', { name: 'computer_use' })

    expect(cu()?.phase).toBe('drafting')
  })

  it('enters running state with target details on tool.start', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Chrome', element: 8 },
      name: 'computer_use',
      tool_id: 'call-cu-1'
    })

    expect(cu()).toMatchObject({ action: 'click', app: 'Chrome', element: 8, phase: 'running', toolId: 'call-cu-1' })
  })

  it("does not carry the previous call's target into the next one", () => {
    mountStream()

    emit('tool.start', { args: { action: 'click', element: 8 }, name: 'computer_use', tool_id: 'call-a' })
    emit('tool.complete', { name: 'computer_use', result: 'Clicked', tool_id: 'call-a' })
    emit('tool.start', { args: { action: 'click', coordinate: [10, 20] }, name: 'computer_use', tool_id: 'call-b' })

    expect(cu()).toMatchObject({ coordinate: [10, 20], phase: 'running', toolId: 'call-b' })
    expect(cu()?.element).toBeUndefined()
    expect(cu()?.completedAt).toBeUndefined()
  })

  it('keeps typed text out of the store', () => {
    mountStream()

    emit('tool.start', { args: { action: 'type', text: 'hunter2' }, name: 'computer_use', tool_id: 'call-type' })

    expect(cu()).toMatchObject({ action: 'type', textLength: 7 })
    expect(JSON.stringify($computerUseBySession.get())).not.toContain('hunter2')
  })

  it('marks completed on successful tool.complete and auto-dismisses', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Terminal', element: 1 },
      name: 'computer_use',
      tool_id: 'call-cu-2'
    })

    emit('tool.complete', {
      duration_s: 0.25,
      name: 'computer_use',
      result: 'Clicked element 1',
      tool_id: 'call-cu-2'
    })

    expect(cu()?.phase).toBe('completed')
    expect(cu()?.durationSeconds).toBe(0.25)

    vi.advanceTimersByTime(1000)
    expect(cu()).toBeDefined()

    vi.advanceTimersByTime(COMPLETED_DISMISS_DELAY_MS - 1000)
    expect(cu()).toBeUndefined()
  })

  it('detects error from payload.error on tool.complete', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Calculator', element: 99 },
      name: 'computer_use',
      tool_id: 'call-cu-3'
    })

    emit('tool.complete', { error: 'Element 99 out of bounds', name: 'computer_use', tool_id: 'call-cu-3' })

    expect(cu()).toMatchObject({ error: 'Element 99 out of bounds', phase: 'error' })
  })

  it('detects error embedded in tool result string on tool.complete', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Calculator', element: 99 },
      name: 'computer_use',
      tool_id: 'call-cu-3b'
    })

    emit('tool.complete', {
      name: 'computer_use',
      result: JSON.stringify({ error: 'Target window minimized' }),
      tool_id: 'call-cu-3b'
    })

    expect(cu()).toMatchObject({ error: 'Target window minimized', phase: 'error' })
  })

  it('normalizes edge-case errors from ok: false and boolean flags', () => {
    mountStream()

    emit('tool.start', { args: { action: 'click', app: 'Settings' }, name: 'computer_use', tool_id: 'call-err-1' })
    emit('tool.complete', {
      name: 'computer_use',
      result: { hint: 'Window handle invalid', ok: false },
      tool_id: 'call-err-1'
    })
    expect(cu()).toMatchObject({ error: 'Window handle invalid', phase: 'error' })

    emit('tool.start', { args: { action: 'key', keys: 'Ctrl+S' }, name: 'computer_use', tool_id: 'call-err-2' })
    emit('tool.complete', {
      is_error: true,
      message: 'Failed to send keys to focused element',
      name: 'computer_use',
      tool_id: 'call-err-2'
    })
    expect(cu()).toMatchObject({ error: 'Failed to send keys to focused element', phase: 'error' })
  })

  it('preserves completed linger state when message.complete finishes the turn', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'capture', app: 'Chrome', mode: 'vision' },
      name: 'computer_use',
      tool_id: 'call-cu-4'
    })
    emit('tool.complete', { name: 'computer_use', result: 'Captured screen', tool_id: 'call-cu-4' })
    expect(cu()?.phase).toBe('completed')

    emit('message.complete', { text: 'All operations finished.' })
    expect(cu()?.phase).toBe('completed')

    vi.advanceTimersByTime(COMPLETED_DISMISS_DELAY_MS)
    expect(cu()).toBeUndefined()
  })

  it('clears unfinished running state when message.complete finishes turn', () => {
    mountStream()

    emit('tool.start', { args: { action: 'capture', app: 'Chrome' }, name: 'computer_use', tool_id: 'call-cu-open' })
    expect(cu()?.phase).toBe('running')

    emit('message.complete', { text: 'Interrupted turn' })

    expect(cu()).toBeUndefined()
  })

  it('isolates events by session', () => {
    mountStream()

    emit(
      'tool.start',
      { args: { action: 'type', app: 'Slack' }, name: 'computer_use', tool_id: 'call-cu-5' },
      OTHER_SID
    )

    expect(cu()).toBeUndefined()
    expect($computerUseBySession.get()[OTHER_SID]?.app).toBe('Slack')
  })

  it('cancels drafting state when another tool starts instead of computer_use', () => {
    mountStream()

    emit('tool.generating', { name: 'computer_use' })
    expect(cu()?.phase).toBe('drafting')

    emit('tool.start', { args: { command: 'ls -la' }, name: 'bash', tool_id: 'call-bash-1' })

    expect(cu()).toBeUndefined()
  })

  it('ignores completions from other tools', () => {
    mountStream()

    emit('tool.start', { args: { action: 'type', app: 'Notepad' }, name: 'computer_use', tool_id: 'call-cu-targeted' })

    emit('tool.complete', { result: 'finished other tool', tool_id: 'call-other-tool' })
    emit('tool.complete', { name: 'terminal', result: 'done', tool_id: 'call-terminal' })

    expect(cu()).toMatchObject({ phase: 'running', toolId: 'call-cu-targeted' })
  })

  it('prevents stale tool.complete from completing a newer active call (A start -> B start -> A complete -> B remains running)', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Chrome', element: 1 },
      name: 'computer_use',
      tool_id: 'call-a'
    })
    emit('tool.start', {
      args: { action: 'type', app: 'Terminal', text: 'npm test' },
      name: 'computer_use',
      tool_id: 'call-b'
    })
    expect(cu()).toMatchObject({ app: 'Terminal', phase: 'running', toolId: 'call-b' })

    emit('tool.complete', { duration_s: 0.15, name: 'computer_use', result: 'Clicked element 1', tool_id: 'call-a' })
    expect(cu()).toMatchObject({ app: 'Terminal', phase: 'running', toolId: 'call-b' })

    emit('tool.complete', { duration_s: 0.45, name: 'computer_use', result: 'Typed text', tool_id: 'call-b' })
    expect(cu()).toMatchObject({ durationSeconds: 0.45, phase: 'completed' })
  })

  describe('after Stop', () => {
    it('ignores a late tool.start from the stopped turn', () => {
      mountStream()

      emit('tool.start', { args: { action: 'capture' }, name: 'computer_use', tool_id: 'call-a' })
      stop()
      emit('tool.start', { args: { action: 'capture' }, name: 'computer_use', tool_id: 'call-a' })

      expect(cu()).toBeUndefined()
    })

    it('ignores a late error completion from the stopped turn', () => {
      mountStream()

      emit('tool.start', { args: { action: 'click' }, name: 'computer_use', tool_id: 'call-a' })
      stop()
      emit('tool.complete', { error: 'Error: canceled', name: 'computer_use', tool_id: 'call-a' })

      expect(cu()).toBeUndefined()
    })

    it('ignores a late successful completion from the stopped turn', () => {
      mountStream()

      emit('tool.start', { args: { action: 'click' }, name: 'computer_use', tool_id: 'call-a' })
      stop()
      emit('tool.complete', { name: 'computer_use', result: 'Clicked', tool_id: 'call-a' })

      expect(cu()).toBeUndefined()
    })
  })
})
