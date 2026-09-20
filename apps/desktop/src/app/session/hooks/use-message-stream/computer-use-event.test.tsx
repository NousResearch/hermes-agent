import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import {
  $computerUseBySession,
  clearAllComputerUseStates,
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

function emit(type: GatewayEvent['type'] | string, payload: any = {}, sessionId = SID) {
  act(() => stream.handleEvent({ payload, session_id: sessionId, type: type as GatewayEvent['type'] }))
}

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

    const state = $computerUseBySession.get()[SID]
    expect(state).toBeDefined()
    expect(state?.phase).toBe('drafting')
  })

  it('enters running state with target details on tool.start', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Chrome', element: 8 },
      name: 'computer_use',
      tool_id: 'call-cu-1'
    })

    const state = $computerUseBySession.get()[SID]
    expect(state).toBeDefined()
    expect(state?.phase).toBe('running')
    expect(state?.app).toBe('Chrome')
    expect(state?.action).toBe('click')
    expect(state?.targetSummary).toBe('Chrome · Click #8')
  })

  it('updates state on tool.progress without clobbering app or action', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'type', app: 'Notepad' },
      name: 'computer_use',
      tool_id: 'call-cu-1'
    })

    expect($computerUseBySession.get()[SID]?.targetSummary).toBe('Notepad · Type')

    // tool.progress with only typing text (no app re-sent)
    emit('tool.progress', {
      args: { text: 'saving document' },
      name: 'computer_use',
      tool_id: 'call-cu-1'
    })

    // Retains Notepad and type action
    const state = $computerUseBySession.get()[SID]
    expect(state?.app).toBe('Notepad')
    expect(state?.action).toBe('type')
    expect(state?.targetSummary).toBe('Notepad · Type "saving docum…"')
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

    const state = $computerUseBySession.get()[SID]
    expect(state?.phase).toBe('completed')
    expect(state?.durationSeconds).toBe(0.25)

    // Linger
    vi.advanceTimersByTime(1000)
    expect($computerUseBySession.get()[SID]).toBeDefined()

    // Dismisses after full delay
    vi.advanceTimersByTime(COMPLETED_DISMISS_DELAY_MS - 1000)
    expect($computerUseBySession.get()[SID]).toBeUndefined()
  })

  it('detects error from payload.error on tool.complete', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Calculator', element: 99 },
      name: 'computer_use',
      tool_id: 'call-cu-3'
    })

    emit('tool.complete', {
      error: 'Element 99 out of bounds',
      name: 'computer_use',
      tool_id: 'call-cu-3'
    })

    const state = $computerUseBySession.get()[SID]
    expect(state?.phase).toBe('error')
    expect(state?.error).toBe('Element 99 out of bounds')
  })

  it('detects error embedded in tool result string on tool.complete', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Calculator', element: 99 },
      name: 'computer_use',
      tool_id: 'call-cu-3b'
    })

    // Result is a JSON string with error payload from Python
    emit('tool.complete', {
      name: 'computer_use',
      result: JSON.stringify({ error: 'Target window minimized' }),
      tool_id: 'call-cu-3b'
    })

    const state = $computerUseBySession.get()[SID]
    expect(state?.phase).toBe('error')
    expect(state?.error).toBe('Target window minimized')
  })

  it('preserves completed linger state when message.complete finishes the turn', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'capture', app: 'Chrome', mode: 'vision' },
      name: 'computer_use',
      tool_id: 'call-cu-4'
    })

    emit('tool.complete', {
      name: 'computer_use',
      result: 'Captured screen',
      tool_id: 'call-cu-4'
    })

    expect($computerUseBySession.get()[SID]?.phase).toBe('completed')

    // Turn ends
    emit('message.complete', { text: 'All operations finished.' })

    // Completed state lingers for user feedback rather than vanishing instantly
    expect($computerUseBySession.get()[SID]?.phase).toBe('completed')

    // After timer, it cleanly dismisses
    vi.advanceTimersByTime(COMPLETED_DISMISS_DELAY_MS)
    expect($computerUseBySession.get()[SID]).toBeUndefined()
  })

  it('clears unfinished running state when message.complete finishes turn', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'capture', app: 'Chrome' },
      name: 'computer_use',
      tool_id: 'call-cu-incomplete'
    })

    expect($computerUseBySession.get()[SID]?.phase).toBe('running')

    // Turn ended unexpectedly without tool.complete
    emit('message.complete', { text: 'Interrupted turn' })

    // Unfinished running state is cleared so it does not stay pinned
    expect($computerUseBySession.get()[SID]).toBeUndefined()
  })

  it('isolates events by session', () => {
    mountStream()

    emit(
      'tool.start',
      {
        args: { action: 'type', app: 'Slack' },
        name: 'computer_use',
        tool_id: 'call-cu-5'
      },
      OTHER_SID
    )

    expect($computerUseBySession.get()[SID]).toBeUndefined()
    expect($computerUseBySession.get()[OTHER_SID]?.app).toBe('Slack')
  })

  it('handles anonymous tool.progress events when computer_use is active', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Firefox', element: 3 },
      name: 'computer_use',
      tool_id: 'call-anon-1'
    })

    expect($computerUseBySession.get()[SID]?.targetSummary).toBe('Firefox · Click #3')

    // Anonymous progress event without name: 'computer_use'
    emit('tool.progress', {
      args: { element: 7 },
      tool_id: 'call-anon-1'
    })

    // State updates correctly using active session context
    expect($computerUseBySession.get()[SID]?.targetSummary).toBe('Firefox · Click #7')
  })

  it('safeguards against rapid progress events arriving after tool.complete', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'click', app: 'Excel' },
      name: 'computer_use',
      tool_id: 'call-rapid-1'
    })

    emit('tool.complete', {
      duration_s: 0.1,
      name: 'computer_use',
      result: 'Clicked',
      tool_id: 'call-rapid-1'
    })

    expect($computerUseBySession.get()[SID]?.phase).toBe('completed')

    // Rapid progress arrives late
    emit('tool.progress', {
      args: { action: 'click', app: 'Excel' },
      name: 'computer_use',
      tool_id: 'call-rapid-1'
    })

    // Phase must remain completed, not reset to running
    expect($computerUseBySession.get()[SID]?.phase).toBe('completed')
  })

  it('normalizes edge-case errors from ok: false, exit_code, and boolean flags', () => {
    mountStream()

    // 1. ok: false with hint in object
    emit('tool.start', {
      args: { action: 'click', app: 'Settings' },
      name: 'computer_use',
      tool_id: 'call-err-1'
    })
    emit('tool.complete', {
      name: 'computer_use',
      result: { hint: 'Window handle invalid', ok: false },
      tool_id: 'call-err-1'
    })
    expect($computerUseBySession.get()[SID]?.phase).toBe('error')
    expect($computerUseBySession.get()[SID]?.error).toBe('Window handle invalid')

    // 2. is_error: true flag
    emit('tool.start', {
      args: { action: 'key', keys: 'Ctrl+S' },
      name: 'computer_use',
      tool_id: 'call-err-2'
    })
    emit('tool.complete', {
      is_error: true,
      message: 'Failed to send keys to focused element',
      name: 'computer_use',
      tool_id: 'call-err-2'
    })
    expect($computerUseBySession.get()[SID]?.phase).toBe('error')
    expect($computerUseBySession.get()[SID]?.error).toBe('Failed to send keys to focused element')

    // 3. Anonymous tool.complete for active computer_use
    emit('tool.start', {
      args: { action: 'wait' },
      name: 'computer_use',
      tool_id: 'call-err-3'
    })
    expect($computerUseBySession.get()[SID]?.phase).toBe('running')
    emit('tool.complete', {
      duration_s: 0.5,
      result: 'Waited 0.5s',
      tool_id: 'call-err-3'
    })
    expect($computerUseBySession.get()[SID]?.phase).toBe('completed')
  })

  it('cancels drafting state when another tool starts instead of computer_use', () => {
    mountStream()

    // Model starts generating computer_use
    emit('tool.generating', { name: 'computer_use' })
    expect($computerUseBySession.get()[SID]?.phase).toBe('drafting')

    // Model actually emits tool.start for bash instead
    emit('tool.start', {
      args: { command: 'ls -la' },
      name: 'bash',
      tool_id: 'call-bash-1'
    })

    // Drafting state must be cleared
    expect($computerUseBySession.get()[SID]).toBeUndefined()
  })

  it('ignores anonymous progress and complete events from other tools with different tool_id', () => {
    mountStream()

    emit('tool.start', {
      args: { action: 'type', app: 'Notepad' },
      name: 'computer_use',
      tool_id: 'call-cu-targeted'
    })
    expect($computerUseBySession.get()[SID]?.phase).toBe('running')

    // Anonymous progress from a different tool call
    emit('tool.progress', {
      args: { text: 'different tool output' },
      tool_id: 'call-other-tool'
    })
    // Computer use state targetSummary should NOT be overwritten by the other tool
    expect($computerUseBySession.get()[SID]?.targetSummary).toBe('Notepad · Type')

    // Anonymous complete from a different tool call
    emit('tool.complete', {
      result: 'finished other tool',
      tool_id: 'call-other-tool'
    })
    // Computer use state must STILL be running, not marked completed prematurely
    expect($computerUseBySession.get()[SID]?.phase).toBe('running')
  })

  it('prevents stale tool.complete from completing a newer active call (Call A start -> Call B start -> Call A complete -> Call B remains running)', () => {
    mountStream()

    // 1. Call A starts
    emit('tool.start', {
      args: { action: 'click', app: 'Chrome', element: 1 },
      name: 'computer_use',
      tool_id: 'call-a'
    })
    expect($computerUseBySession.get()[SID]?.phase).toBe('running')
    expect($computerUseBySession.get()[SID]?.toolId).toBe('call-a')
    expect($computerUseBySession.get()[SID]?.app).toBe('Chrome')

    // 2. Call B starts
    emit('tool.start', {
      args: { action: 'type', app: 'Terminal', text: 'npm test' },
      name: 'computer_use',
      tool_id: 'call-b'
    })
    expect($computerUseBySession.get()[SID]?.phase).toBe('running')
    expect($computerUseBySession.get()[SID]?.toolId).toBe('call-b')
    expect($computerUseBySession.get()[SID]?.app).toBe('Terminal')

    // 3. Stale Call A named complete arrives
    emit('tool.complete', {
      duration_s: 0.15,
      name: 'computer_use',
      result: 'Clicked element 1',
      tool_id: 'call-a'
    })

    // Call B must REMAIN running, not completed by Call A
    const activeState = $computerUseBySession.get()[SID]
    expect(activeState?.phase).toBe('running')
    expect(activeState?.toolId).toBe('call-b')
    expect(activeState?.app).toBe('Terminal')

    // 4. Call B named complete arrives
    emit('tool.complete', {
      duration_s: 0.45,
      name: 'computer_use',
      result: 'Typed text',
      tool_id: 'call-b'
    })

    // Now it completes successfully
    expect($computerUseBySession.get()[SID]?.phase).toBe('completed')
    expect($computerUseBySession.get()[SID]?.durationSeconds).toBe(0.45)
  })
})

