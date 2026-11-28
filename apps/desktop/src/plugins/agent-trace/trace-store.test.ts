/**
 * Pure-reducer tests — synthetic gateway frames drive `applyEvent` and the
 * snapshot merges exactly like the live tap does. No React, no gateway.
 */

import { beforeEach, describe, expect, it } from 'vitest'

import {
  $selectedStep,
  $trace,
  $traceFilter,
  $tracePaused,
  $traceScope,
  activeSubagentCount,
  applyEvent,
  detectToolFailure,
  EMPTY_TRACE,
  findTool,
  hydrateHistory,
  mergeDelegationStatus,
  mergeSubagentList,
  runningToolCount,
  sessionById
} from './trace-store'
import type { TraceEvent, TraceState } from './types'

const T0 = 1_700_000_000

function ev(type: string, payload: unknown, overrides: Partial<TraceEvent> = {}): TraceEvent {
  return { type, at: T0, sessionId: 's1', profile: 'default', payload, ...overrides }
}

function toolStart(id: string, name: string, args: Record<string, unknown> = {}, at = T0): TraceEvent {
  return ev('tool.start', { tool_id: id, name, args, preview: `${name} preview` }, { at })
}

function toolComplete(id: string, name: string, extra: Record<string, unknown> = {}, at = T0 + 2): TraceEvent {
  return ev('tool.complete', { tool_id: id, name, duration_s: 2, result_text: 'ok', ...extra }, { at })
}

beforeEach(() => {
  $trace.set(EMPTY_TRACE)
  $selectedStep.set(null)
  $traceFilter.set('')
  $traceScope.set('focused')
  $tracePaused.set(false)
})

describe('detectToolFailure', () => {
  it('flags success:false with an error string', () => {
    expect(detectToolFailure({ success: false, error: 'boom' })).toEqual({ failed: true, error: 'boom' })
  })

  it('flags ok:false', () => {
    expect(detectToolFailure({ ok: false }).failed).toBe(true)
  })

  it('flags a non-zero exit_code even without an error string', () => {
    const result = detectToolFailure({ output: '', exit_code: 1, error: null })
    expect(result.failed).toBe(true)
    expect(result.error).toContain('exit')
  })

  it('parses a JSON result_text when result is absent', () => {
    expect(detectToolFailure(undefined, '{"success": false, "error": "denied"}')).toEqual({
      failed: true,
      error: 'denied'
    })
  })

  it('passes clean results and non-JSON text', () => {
    expect(detectToolFailure({ success: true }).failed).toBe(false)
    expect(detectToolFailure(undefined, 'plain output').failed).toBe(false)
    expect(detectToolFailure({ exit_code: 0 }).failed).toBe(false)
  })
})

describe('tool lifecycle', () => {
  it('start → complete carries args, latency and output', () => {
    let state = applyEvent(EMPTY_TRACE, toolStart('t1', 'read_file', { path: '/tmp/x' }))
    const running = sessionById(state, 's1')
    expect(running?.tools[0]).toMatchObject({ id: 't1', name: 'read_file', status: 'running', origin: 'live' })
    expect(running?.tools[0].args).toEqual({ path: '/tmp/x' })
    expect(runningToolCount(state)).toBe(1)

    state = applyEvent(state, toolComplete('t1', 'read_file', { result_text: 'file body' }))
    const done = sessionById(state, 's1')?.tools[0]
    expect(done).toMatchObject({ status: 'done', durationS: 2, resultText: 'file body', resultChars: 9 })
    expect(done?.endedAt).toBe(T0 + 2)
    expect(runningToolCount(state)).toBe(0)
  })

  it('computes latency from timestamps when duration_s is missing', () => {
    let state = applyEvent(EMPTY_TRACE, toolStart('t1', 'terminal', {}, T0))
    state = applyEvent(
      state,
      ev('tool.complete', { tool_id: 't1', name: 'terminal', result_text: 'ok' }, { at: T0 + 3.5 })
    )
    expect(sessionById(state, 's1')?.tools[0].durationS).toBeCloseTo(3.5)
  })

  it('marks failed calls as error and keeps the error trace', () => {
    let state = applyEvent(EMPTY_TRACE, toolStart('t1', 'terminal'))
    state = applyEvent(
      state,
      toolComplete('t1', 'terminal', {
        result: { output: '', exit_code: 2, error: null },
        result_text: '{"output":"","exit_code":2}'
      })
    )
    const tool = sessionById(state, 's1')?.tools[0]
    expect(tool?.status).toBe('error')
    expect(tool?.error).toContain('2')
  })

  it('generating placeholder is cleared by tool.start', () => {
    let state = applyEvent(EMPTY_TRACE, ev('tool.generating', { name: 'web_search' }))
    expect(sessionById(state, 's1')?.generating).toMatchObject({ name: 'web_search' })
    state = applyEvent(state, toolStart('t1', 'web_search'))
    expect(sessionById(state, 's1')?.generating).toBeNull()
  })

  it('progress updates a running call', () => {
    let state = applyEvent(EMPTY_TRACE, toolStart('t1', 'browser_exec'))
    state = applyEvent(
      state,
      ev('tool.progress', { tool_id: 't1', name: 'browser_exec', progress: 3, total: 10, message: 'step' })
    )
    expect(sessionById(state, 's1')?.tools[0].progress).toEqual({ value: 3, total: 10, message: 'step' })
  })

  it('ignores frames without a session id', () => {
    const state = applyEvent(EMPTY_TRACE, ev('tool.start', { tool_id: 't', name: 'x' }, { sessionId: null }))
    expect(state).toBe(EMPTY_TRACE)
  })
})

describe('subagent tree', () => {
  const spawn = (id: string, extra: Record<string, unknown> = {}, at = T0) =>
    ev(
      'subagent.spawn_requested',
      { goal: 'research X', task_count: 2, task_index: 0, subagent_id: id, ...extra },
      { at }
    )

  it('spawn → start → tool → complete builds one node with rollups', () => {
    let state = applyEvent(EMPTY_TRACE, spawn('a1', { delegation_id: 'd1', depth: 0 }))
    expect(sessionById(state, 's1')?.subagents[0]).toMatchObject({ id: 'a1', status: 'queued', goal: 'research X' })

    state = applyEvent(
      state,
      ev(
        'subagent.start',
        { goal: 'research X', task_count: 2, task_index: 0, subagent_id: 'a1', model: 'nous/qwen4-max' },
        { at: T0 + 1 }
      )
    )
    expect(sessionById(state, 's1')?.subagents[0]).toMatchObject({ status: 'running', model: 'nous/qwen4-max' })
    expect(activeSubagentCount(state)).toBe(1)

    state = applyEvent(
      state,
      ev(
        'subagent.tool',
        {
          goal: 'research X',
          task_count: 2,
          task_index: 0,
          subagent_id: 'a1',
          tool_name: 'web_search',
          tool_preview: 'q…',
          tool_count: 1
        },
        { at: T0 + 2 }
      )
    )
    const node = sessionById(state, 's1')?.subagents[0]
    expect(node?.lastTool).toBe('web_search')
    expect(node?.tools[0]).toMatchObject({ name: 'web_search', subagentId: 'a1' })

    state = applyEvent(
      state,
      ev(
        'subagent.complete',
        {
          goal: 'research X',
          task_count: 2,
          task_index: 0,
          subagent_id: 'a1',
          status: 'completed',
          duration_seconds: 12,
          summary: 'done',
          input_tokens: 1000,
          output_tokens: 200,
          api_calls: 3,
          output_tail: [{ tool: 'web_search', preview: 'result', is_error: false }]
        },
        { at: T0 + 13 }
      )
    )
    const finished = sessionById(state, 's1')?.subagents[0]
    expect(finished).toMatchObject({
      status: 'completed',
      durationS: 12,
      inputTokens: 1000,
      outputTokens: 200,
      apiCalls: 3,
      summary: 'done'
    })
    expect(finished?.outputTail).toHaveLength(1)
    expect(activeSubagentCount(state)).toBe(0)
  })

  it('keeps parent links for nested delegation', () => {
    let state = applyEvent(EMPTY_TRACE, spawn('a1'))
    state = applyEvent(state, spawn('a2', { parent_id: 'a1', depth: 1, task_index: 1 }))
    const nodes = sessionById(state, 's1')?.subagents ?? []
    expect(nodes.find(n => n.id === 'a2')).toMatchObject({ parentId: 'a1', depth: 1 })
  })

  it('failed completion keeps the error summary', () => {
    let state = applyEvent(EMPTY_TRACE, spawn('a1'))
    state = applyEvent(
      state,
      ev('subagent.complete', {
        goal: 'research X',
        task_count: 1,
        task_index: 0,
        subagent_id: 'a1',
        status: 'failed',
        summary: 'model error'
      })
    )
    expect(sessionById(state, 's1')?.subagents[0]).toMatchObject({ status: 'failed', error: 'model error' })
  })

  it('delegation.status merges under the owner session and never reopens terminal nodes', () => {
    let state = applyEvent(EMPTY_TRACE, spawn('a1'))
    state = applyEvent(
      state,
      ev('subagent.complete', {
        goal: 'research X',
        task_count: 1,
        task_index: 0,
        subagent_id: 'a1',
        status: 'completed'
      })
    )
    state = mergeDelegationStatus(
      state,
      {
        active: [
          { subagent_id: 'a1', owner_agent_session_id: 's1', status: 'running', goal: 'research X' },
          {
            subagent_id: 'a2',
            owner_agent_session_id: 's2',
            status: 'running',
            goal: 'other',
            depth: 0,
            started_at: T0
          }
        ],
        paused: false,
        max_spawn_depth: 2,
        max_concurrent_children: 5
      },
      T0 + 5
    )
    expect(sessionById(state, 's1')?.subagents[0].status).toBe('completed')
    expect(sessionById(state, 's2')?.subagents[0]).toMatchObject({ id: 'a2', status: 'running' })
  })

  it('subagent.list adds failed async delegations as failed nodes', () => {
    const state = mergeSubagentList(
      EMPTY_TRACE,
      's1',
      {
        subagents: [],
        delegations: [
          { delegation_id: 'd9', task_index: 1, status: 'failed', goal: 'build Y', error: 'timeout waiting' }
        ]
      },
      T0
    )

    const node = sessionById(state, 's1')?.subagents[0]
    expect(node).toMatchObject({ status: 'failed', goal: 'build Y', error: 'timeout waiting' })
  })
})

describe('session frames', () => {
  it('message.start/complete toggle turn activity and merge usage', () => {
    let state = applyEvent(EMPTY_TRACE, ev('message.start', {}))
    expect(sessionById(state, 's1')?.turnActive).toBe(true)
    state = applyEvent(
      state,
      ev('message.complete', { usage: { input: 100, output: 50, context_used: 150, context_max: 1000 } })
    )
    const session = sessionById(state, 's1')
    expect(session?.turnActive).toBe(false)
    expect(session?.usage).toMatchObject({ input: 100, output: 50, context_used: 150 })
  })

  it('session.usage folds deprecated top-level counters', () => {
    const state = applyEvent(
      EMPTY_TRACE,
      ev('session.usage', {
        context_tokens: 900,
        context_length: 2000,
        input_tokens: 800,
        output_tokens: 100,
        model: 'm'
      })
    )

    expect(sessionById(state, 's1')?.usage).toMatchObject({
      context_used: 900,
      context_max: 2000,
      input: 800,
      output: 100,
      model: 'm'
    })
  })

  it('error frames surface as lastError', () => {
    const state = applyEvent(EMPTY_TRACE, ev('error', { message: 'agent init failed' }))
    expect(sessionById(state, 's1')?.lastError).toMatchObject({ message: 'agent init failed' })
  })
})

describe('history hydration', () => {
  it('fills persisted tool rows but never overwrites live calls', () => {
    let state = applyEvent(EMPTY_TRACE, toolStart('t1', 'read_file', { path: '/live' }))
    state = applyEvent(state, toolComplete('t1', 'read_file', { result_text: 'live output' }))
    state = hydrateHistory(
      state,
      's1',
      [
        { role: 'tool', tool_call_id: 't1', name: 'read_file', args: { path: '/stale' }, timestamp: T0 - 5 },
        {
          role: 'tool',
          tool_call_id: 't0',
          name: 'terminal',
          args: { command: 'ls' },
          context: 'ls preview',
          timestamp: T0 - 10
        },
        { role: 'assistant', text: 'hello' }
      ],
      T0
    )
    const session = sessionById(state, 's1')
    expect(session?.tools).toHaveLength(2)
    expect(session?.tools.find(t => t.id === 't1')).toMatchObject({ origin: 'live', resultText: 'live output' })
    expect(session?.tools.find(t => t.id === 't0')).toMatchObject({
      origin: 'history',
      name: 'terminal',
      preview: 'ls preview',
      endedAt: T0 - 10
    })
    expect(session?.historyHydrated).toBe(true)
  })

  it('findTool resolves session and child rows', () => {
    let state = applyEvent(EMPTY_TRACE, toolStart('t1', 'read_file'))
    expect(findTool(state, { kind: 'tool', sessionId: 's1', toolId: 't1' })?.name).toBe('read_file')
    state = applyEvent(
      state,
      ev('subagent.tool', {
        goal: 'g',
        task_count: 1,
        task_index: 0,
        subagent_id: 'a1',
        tool_name: 'web_search',
        tool_count: 1
      })
    )
    const childToolId = sessionById(state, 's1')?.subagents[0].tools[0].id as string
    expect(findTool(state, { kind: 'tool', sessionId: 's1', toolId: childToolId, subagentId: 'a1' })?.name).toBe(
      'web_search'
    )
  })
})

describe('memory bounds', () => {
  it('prunes the least-recently-active sessions past the cap', () => {
    let state: TraceState = EMPTY_TRACE

    for (let i = 0; i < 30; i++) {
      state = applyEvent(state, ev('message.start', {}, { sessionId: `s${i}`, at: T0 + i }))
    }

    expect(state.sessions.length).toBeLessThanOrEqual(24)
    // The most recent frames survive; the oldest session was evicted.
    expect(sessionById(state, 's29')).not.toBeNull()
    expect(sessionById(state, 's0')).toBeNull()
  })
})
