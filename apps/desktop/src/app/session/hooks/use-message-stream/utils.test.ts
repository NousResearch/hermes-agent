import { describe, expect, it } from 'vitest'

import { sessionInfoPayload, toolCompletePayload, toolStartPayload } from '@/test/contract'

import {
  completionErrorText,
  delegateTaskPayloads,
  hasSessionInfoStatePatch,
  sessionInfoStatePatch,
  toTodoPayload
} from './utils'

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
  it('routes todo-named events to the todo stream and leaves other tools alone', () => {
    expect(toTodoPayload(toolStartPayload({ name: 'todo' }))?.tool_id).toBe('todo-live')
    expect(toTodoPayload(toolCompletePayload({ name: 'todo', todos: [] }))?.name).toBe('todo_list')
    expect(toTodoPayload(toolStartPayload({ name: 'todo_list' }))?.tool_id).toBe('todo-live')
    expect(toTodoPayload(toolCompletePayload({ name: 'todo_list', tool_id: 'call-1' }))?.tool_id).toBe('call-1')
    expect(toTodoPayload(toolStartPayload({ name: 'web_search' }))).toBeUndefined()
    expect(toTodoPayload(toolCompletePayload({ name: 'terminal', todos: [] }))).toBeUndefined()
  })
})

describe('sessionInfoStatePatch / hasSessionInfoStatePatch', () => {
  it('extracts only present runtime fields', () => {
    const patch = sessionInfoStatePatch(sessionInfoPayload({ model: 'gpt', fast: true, branch: 'main' }))
    expect(patch).toMatchObject({ model: 'gpt', fast: true, branch: 'main' })
    expect(hasSessionInfoStatePatch(patch)).toBe(true)
    expect(hasSessionInfoStatePatch(sessionInfoStatePatch(undefined))).toBe(false)
  })
})

describe('delegateTaskPayloads', () => {
  it('returns [] for non-delegate events', () => {
    expect(delegateTaskPayloads(toolStartPayload({ name: 'web_search' }), 'running')).toEqual([])
  })

  it('maps a running tool.start to a subagent.start spec', () => {
    const [spec] = delegateTaskPayloads(
      toolStartPayload({ name: 'delegate_task', tool_id: 't1', args: { goal: 'do it' } }),
      'running',
      'tool.start'
    )

    expect(spec).toMatchObject({ goal: 'do it', status: 'running', tool_name: 'delegate_task', text: null })
  })

  it('maps completion (with error) to a failed subagent.complete', () => {
    const [spec] = delegateTaskPayloads(
      toolCompletePayload({ name: 'delegate_task', result: { error: 'boom', summary: 'failed run' } }),
      'complete'
    )

    expect(spec).toMatchObject({ status: 'failed', tool_name: null })
  })

  it.each(['timeout', 'error', 'failed', 'failure', 'TIMEOUT'])(
    'maps completion with result.status=%s to a failed subagent.complete',
    resultStatus => {
      const [spec] = delegateTaskPayloads(
        toolCompletePayload({ name: 'delegate_task', result: { status: resultStatus, summary: 'timed out' } }),
        'complete'
      )

      expect(spec).toMatchObject({ status: 'failed', tool_name: null })
    }
  )

  it('maps a successful completion to completed', () => {
    const [spec] = delegateTaskPayloads(
      toolCompletePayload({ name: 'delegate_task', result: { status: 'success', summary: 'done' } }),
      'complete'
    )

    expect(spec).toMatchObject({ status: 'completed', summary: 'done', tool_name: null })
  })

  it('reads spend and the classified reason off the child s own result entry', () => {
    const [spec] = delegateTaskPayloads(
      toolCompletePayload({
        name: 'delegate_task',
        args: { tasks: [{ goal: 'do it' }] },
        result: { results: [{ cost_usd: 0.5, failure_reason: 'timeout', status: 'failed', summary: 'timed out' }] }
      }),
      'complete'
    )

    expect(spec).toMatchObject({ cost_usd: 0.5, failure_reason: 'timeout' })
  })

  it('falls back to a flat result for a single-task payload', () => {
    const [spec] = delegateTaskPayloads(
      toolCompletePayload({ name: 'delegate_task', result: { cost_usd: 0.25, failure_reason: 'billing', status: 'failed' } }),
      'complete'
    )

    expect(spec).toMatchObject({ cost_usd: 0.25, failure_reason: 'billing' })
  })
})
