import { describe, expect, it } from 'vitest'

import type { GatewayEventPayload } from '@/lib/chat-messages'

import {
  completionErrorText,
  delegateTaskPayloads,
  hasSessionInfoStatePatch,
  isStaleCompletion,
  sessionInfoStatePatch,
  toTodoPayload
} from './utils'

const payload = (over: Record<string, unknown>): GatewayEventPayload => over as GatewayEventPayload

describe('isStaleCompletion', () => {
  const idle = { supersededTurnToken: null, turnToken: null }

  it('never treats an absent or empty turn token as stale (older gateways)', () => {
    expect(isStaleCompletion(undefined, { ...idle, turnToken: 'live', supersededTurnToken: 'old' })).toBe(false)
    expect(isStaleCompletion('', { ...idle, turnToken: 'live', supersededTurnToken: 'old' })).toBe(false)
  })

  it('drops a complete whose turn equals the superseded token (seed/rewind arm)', () => {
    expect(isStaleCompletion('old', { supersededTurnToken: 'old', turnToken: null })).toBe(true)
  })

  it('drops a mismatched stamped complete when a live token is already claimed', () => {
    expect(isStaleCompletion('other', { supersededTurnToken: null, turnToken: 'live' })).toBe(true)
  })

  it('keeps the matching stamped complete for the in-flight turn', () => {
    expect(isStaleCompletion('live', { supersededTurnToken: 'old', turnToken: 'live' })).toBe(false)
  })

  it('keeps a stamped complete when no identity is claimed yet (turnToken null)', () => {
    // Seed nulls turnToken and arms superseded; a token that is neither
    // superseded nor a claim-mismatch stays live (muted/backend-originated).
    expect(isStaleCompletion('fresh', { supersededTurnToken: 'old', turnToken: null })).toBe(false)
    expect(isStaleCompletion('fresh', idle)).toBe(false)
  })
})

describe('completionErrorText', () => {
  it('flags provider/HTTP/retry failures, ignores normal text', () => {
    expect(completionErrorText('API call failed after 3 retries: boom')).toMatch(/^API call failed/)
    expect(completionErrorText('HTTP 500 upstream')).toMatch(/^HTTP 500/)
    expect(completionErrorText('Gateway error: nope')).toMatch(/^Gateway error/)
    expect(completionErrorText('here is your answer')).toBeNull()
    expect(completionErrorText('   ')).toBeNull()
  })
})

describe('toTodoPayload', () => {
  it('routes named todo and anonymous todos-bearing events to the todo stream', () => {
    expect(toTodoPayload(payload({ name: 'todo' }))?.tool_id).toBe('todo-live')
    expect(toTodoPayload(payload({ todos: [] }))?.name).toBe('todo_list')
    expect(toTodoPayload(payload({ name: 'todo_list' }))?.tool_id).toBe('todo-live')
    expect(toTodoPayload(payload({ name: 'web_search' }))).toBeUndefined()
    expect(toTodoPayload(undefined)).toBeUndefined()
  })
})

describe('sessionInfoStatePatch / hasSessionInfoStatePatch', () => {
  it('extracts only present runtime fields', () => {
    const patch = sessionInfoStatePatch(payload({ model: 'gpt', fast: true, branch: 'main' }))
    expect(patch).toMatchObject({ model: 'gpt', fast: true, branch: 'main' })
    expect(hasSessionInfoStatePatch(patch)).toBe(true)
    expect(hasSessionInfoStatePatch(sessionInfoStatePatch(payload({})))).toBe(false)
  })
})

describe('delegateTaskPayloads', () => {
  it('returns [] for non-delegate events', () => {
    expect(delegateTaskPayloads(payload({ name: 'web_search' }), 'running')).toEqual([])
  })

  it('maps a running tool.start to a subagent.start spec', () => {
    const [spec] = delegateTaskPayloads(
      payload({ name: 'delegate_task', tool_id: 't1', args: { goal: 'do it' } }),
      'running',
      'tool.start'
    )

    expect(spec).toMatchObject({ event_type: 'subagent.start', goal: 'do it', status: 'running' })
  })

  it('maps completion (with error) to a failed subagent.complete', () => {
    const [spec] = delegateTaskPayloads(
      payload({ name: 'delegate_task', error: 'boom', result: { summary: 'failed run' } }),
      'complete'
    )

    expect(spec).toMatchObject({ event_type: 'subagent.complete', status: 'failed' })
  })

  it.each(['timeout', 'failure', 'TIMEOUT'])(
    'maps completion with result.status=%s to a failed subagent.complete',
    resultStatus => {
      const [spec] = delegateTaskPayloads(
        payload({ name: 'delegate_task', result: { status: resultStatus, summary: 'timed out' } }),
        'complete'
      )

      expect(spec).toMatchObject({ event_type: 'subagent.complete', status: 'failed' })
    }
  )

  it('maps a successful completion to completed', () => {
    const [spec] = delegateTaskPayloads(
      payload({ name: 'delegate_task', result: { status: 'success', summary: 'done' } }),
      'complete'
    )

    expect(spec).toMatchObject({ event_type: 'subagent.complete', status: 'completed' })
  })
})
