import { describe, expect, it } from 'vitest'

import { applyGroupEvent, emptyGroupState, mergeGroupRoom } from './group-model'

describe('group room event reconciliation', () => {
  it('merges parallel agent deltas by message id while preserving attribution', () => {
    let state = mergeGroupRoom(emptyGroupState(), {
      id: 'room-1',
      name: 'Launch',
      profiles: ['planner', 'reviewer'],
      messages: []
    })

    state = applyGroupEvent(state, {
      type: 'group.message.delta',
      payload: { room_id: 'room-1', message_id: 'm1', profile: 'planner', delta: 'Plan ' }
    })
    state = applyGroupEvent(state, {
      type: 'group.message.delta',
      payload: { room_id: 'room-1', message_id: 'm2', profile: 'reviewer', delta: 'Risk' }
    })
    state = applyGroupEvent(state, {
      type: 'group.message.delta',
      payload: { room_id: 'room-1', message_id: 'm1', profile: 'planner', delta: 'ready' }
    })

    expect(state.rooms['room-1'].messages).toEqual([
      expect.objectContaining({ id: 'm1', profile: 'planner', content: 'Plan ready', status: 'streaming' }),
      expect.objectContaining({ id: 'm2', profile: 'reviewer', content: 'Risk', status: 'streaming' })
    ])
  })

  it('appends the backend text field for message.delta while accepting legacy delta', () => {
    let state = applyGroupEvent(emptyGroupState(), {
      type: 'group.message.delta',
      payload: { room_id: 'room-1', message_id: 'm1', profile: 'planner', text: 'Plan ' }
    })

    state = applyGroupEvent(state, {
      type: 'group.message.delta',
      payload: { room_id: 'room-1', message_id: 'm1', profile: 'planner', delta: 'ready' }
    })

    expect(state.rooms['room-1'].messages[0].content).toBe('Plan ready')
  })

  it('deduplicates authoritative message snapshots after live events', () => {
    const live = applyGroupEvent(emptyGroupState(), {
      type: 'group.message.complete',
      payload: { room_id: 'room-1', message_id: 'm1', profile: 'planner', content: 'Done' }
    })

    const merged = mergeGroupRoom(live, {
      id: 'room-1', name: 'Room', profiles: ['planner'],
      messages: [{ id: 'm1', role: 'assistant', profile: 'planner', content: 'Done', status: 'complete' }]
    })

    expect(merged.rooms['room-1'].messages).toHaveLength(1)
    expect(merged.rooms['room-1'].messages[0].status).toBe('complete')
  })

  it('normalizes workspace, compression status, summary, and message sequence', () => {
    const state = mergeGroupRoom(emptyGroupState(), {
      id: 'room-1', name: 'Room', profiles: ['planner'], workspace: '/work/launch',
      summary: 'Earlier decisions were compressed.', context_status: 'compressed',
      messages: [{ id: 'm9', seq: 9, role: 'assistant', content: 'Latest' }]
    })

    expect(state.rooms['room-1']).toEqual(expect.objectContaining({
      workspace: '/work/launch', summary: 'Earlier decisions were compressed.', contextStatus: 'compressed'
    }))
    expect(state.rooms['room-1'].messages[0]).toEqual(expect.objectContaining({ seq: 9 }))
  })

  it('prepends older pages in sequence order and deduplicates the boundary message', () => {
    let state = mergeGroupRoom(emptyGroupState(), {
      id: 'room-1', name: 'Room', profiles: [],
      messages: [{ id: 'm2', seq: 2, content: 'two' }, { id: 'm3', seq: 3, content: 'three' }]
    })

    state = mergeGroupRoom(state, {
      id: 'room-1', name: 'Room', profiles: [],
      messages: [{ id: 'm1', seq: 1, content: 'one' }, { id: 'm2', seq: 2, content: 'two' }]
    })

    expect(state.rooms['room-1'].messages.map(message => message.id)).toEqual(['m1', 'm2', 'm3'])
  })

  it('derives room running from active profiles across completion, error, and interrupt', () => {
    let state = mergeGroupRoom(emptyGroupState(), { id: 'room-1', name: 'Room', profiles: ['planner', 'reviewer'], messages: [] })

    state = applyGroupEvent(state, { type: 'group.run.start', payload: { room_id: 'room-1', profile: 'planner' } })
    state = applyGroupEvent(state, { type: 'group.run.start', payload: { room_id: 'room-1', profile: 'reviewer' } })
    state = applyGroupEvent(state, { type: 'group.run.complete', payload: { room_id: 'room-1', profile: 'planner' } })
    expect(state.rooms['room-1'].running).toBe(true)

    state = applyGroupEvent(state, { type: 'group.message.error', payload: { room_id: 'room-1', profile: 'reviewer', message: 'failed' } })
    expect(state.rooms['room-1'].running).toBe(false)

    state = applyGroupEvent(state, { type: 'group.run.start', payload: { room_id: 'room-1', profile: 'planner' } })
    state = applyGroupEvent(state, { type: 'group.run.start', payload: { room_id: 'room-1', profile: 'reviewer' } })
    state = applyGroupEvent(state, { type: 'group.run.interrupt', payload: { room_id: 'room-1', profile: 'planner' } })
    expect(state.rooms['room-1'].running).toBe(true)
    state = applyGroupEvent(state, { type: 'group.run.interrupt', payload: { room_id: 'room-1' } })
    expect(state.rooms['room-1'].running).toBe(false)
  })

  it('preserves approval command and choice constraints on the owning agent message', () => {
    let state = applyGroupEvent(emptyGroupState(), {
      type: 'group.event',
      payload: { room_id: 'room-1', member_profile: 'planner', type: 'message.start', payload: { message_id: 'm1', session_id: 'runtime-1' } }
    })

    state = applyGroupEvent(state, {
      type: 'group.event',
      payload: {
        room_id: 'room-1', member_profile: 'planner', type: 'approval.request',
        payload: { message_id: 'm1', session_id: 'runtime-1', command: 'rm file', description: 'Delete file', choices: ['once', 'deny'] }
      }
    })

    expect(state.rooms['room-1'].messages[0]).toEqual(expect.objectContaining({
      status: 'approval',
      runtimeSessionId: 'runtime-1',
      approval: expect.objectContaining({
        command: 'rm file',
        description: 'Delete file',
        choices: ['once', 'deny']
      })
    }))
  })

  it('normalizes durable room snapshots with tool and approval details', () => {
    const state = mergeGroupRoom(emptyGroupState(), {
      id: 'room-1',
      name: 'Durable',
      profiles: ['planner'],
      messages: [{
        id: 'm1',
        role: 'assistant',
        profile: 'planner',
        content: '',
        status: 'approval',
        runtime_session_id: 'runtime-1',
        approval: {
          command: 'rm file',
          description: 'Delete file',
          choices: ['once', 'deny'],
          allow_permanent: false,
          smart_denied: true
        },
        tools: [{
          tool_id: 't1', name: 'terminal', context: 'pwd',
          status: 'complete', result: { ok: true }
        }]
      }]
    })

    const message = state.rooms['room-1'].messages[0]

    expect(message.approval).toEqual(expect.objectContaining({
      command: 'rm file', description: 'Delete file', choices: ['once', 'deny'],
      allowPermanent: false, smartDenied: true
    }))
    expect(message.parts[0]).toEqual(expect.objectContaining({
      type: 'tool-call', toolCallId: 't1', toolName: 'terminal', result: expect.objectContaining({ ok: true })
    }))
  })

  it('projects tool lifecycle events into assistant-ui tool-call parts', () => {
    let state = applyGroupEvent(emptyGroupState(), {
      type: 'group.event',
      payload: { room_id: 'room-1', member_profile: 'planner', type: 'message.start', payload: { message_id: 'm1' } }
    })

    state = applyGroupEvent(state, {
      type: 'group.event',
      payload: { room_id: 'room-1', member_profile: 'planner', type: 'tool.start', payload: { tool_id: 't1', name: 'terminal', context: 'npm test' } }
    })
    state = applyGroupEvent(state, {
      type: 'group.event',
      payload: { room_id: 'room-1', member_profile: 'planner', type: 'tool.progress', payload: { tool_id: 't1', preview: 'running' } }
    })
    state = applyGroupEvent(state, {
      type: 'group.event',
      payload: { room_id: 'room-1', member_profile: 'planner', type: 'tool.complete', payload: { tool_id: 't1', name: 'terminal', args: { command: 'npm test' }, result: { success: true } } }
    })

    const part = state.rooms['room-1'].messages[0].parts[0]
    expect(part).toEqual(expect.objectContaining({
      type: 'tool-call',
      toolCallId: 't1',
      toolName: 'terminal',
      result: { success: true }
    }))
    expect(part).toEqual(expect.objectContaining({ args: expect.objectContaining({ command: 'npm test' }) }))
  })
})
