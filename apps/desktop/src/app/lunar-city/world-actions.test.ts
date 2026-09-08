import { describe, expect, it, vi } from 'vitest'

import { createWorldActionRunner } from './world-actions'

function context() {
  return {
    kanban: {
      addComment: vi.fn().mockResolvedValue({}),
      createTask: vi.fn().mockResolvedValue({ task: { id: 'new-task' } }),
      patchTask: vi.fn().mockResolvedValue({}),
      reclaimTask: vi.fn().mockResolvedValue({}),
      reassignTask: vi.fn().mockResolvedValue({})
    },
    respondApproval: vi.fn().mockResolvedValue({ ok: true }),
    sendDialogue: vi.fn().mockResolvedValue({ id: 'reply' }),
    sendVoice: vi.fn().mockResolvedValue({ id: 'voice' })
  }
}

describe('world action runner', () => {
  it('maps extinguish to the existing recovery write', async () => {
    const doors = context()
    const runner = createWorldActionRunner(doors)

    const result = await runner.run({ kind: 'recover_task', patch: { status: 'todo' }, taskId: 'task-7' })

    expect(doors.kanban.patchTask).toHaveBeenCalledWith('task-7', { status: 'todo' })
    expect(result).toEqual({ kind: 'completed', ok: true, value: {} })
  })

  it('does not claim recovery when Hermes rejects the write', async () => {
    const doors = context()
    doors.kanban.patchTask.mockRejectedValue(new Error('409 blocked'))
    const runner = createWorldActionRunner(doors)

    const result = await runner.run({ kind: 'recover_task', patch: { status: 'todo' }, taskId: 'task-7' })

    expect(result).toEqual({ kind: 'rejected', message: '409 blocked', ok: false })
  })

  it('classifies approval and connection failures without bypassing them', async () => {
    const approvalDoors = context()
    approvalDoors.kanban.patchTask.mockRejectedValue(new Error('403 approval required'))
    const disconnectedDoors = context()
    disconnectedDoors.kanban.patchTask.mockRejectedValue(new Error('gateway disconnected'))

    await expect(
      createWorldActionRunner(approvalDoors).run({ kind: 'recover_task', patch: {}, taskId: 'task-7' })
    ).resolves.toMatchObject({
      kind: 'approval_required',
      ok: false
    })
    await expect(
      createWorldActionRunner(disconnectedDoors).run({ kind: 'recover_task', patch: {}, taskId: 'task-7' })
    ).resolves.toMatchObject({
      kind: 'disconnected',
      ok: false
    })
  })

  it('routes task creation and reassignment through existing doors', async () => {
    const doors = context()
    const runner = createWorldActionRunner(doors)

    await runner.run({ body: { assignee: 'worker-a', title: 'New work' }, kind: 'create_task' })
    await runner.run({ kind: 'reassign_task', profile: 'worker-b', taskId: 'task-7' })

    expect(doors.kanban.createTask).toHaveBeenCalledWith({ assignee: 'worker-a', title: 'New work' })
    expect(doors.kanban.reassignTask).toHaveBeenCalledWith('task-7', 'worker-b')
  })

  it('routes in-world dialogue, voice, and approvals through Hermes doors', async () => {
    const doors = context()
    const runner = createWorldActionRunner(doors)

    await runner.run({ kind: 'dialogue_send', message: 'status?', sessionId: 'session-a', targetProfile: 'review' })
    await runner.run({ kind: 'voice_send', sessionId: 'session-a', transcript: 'approve it' })
    await runner.run({ actionId: 'approval-a', approved: true, kind: 'approval_response', sessionId: 'session-a' })

    expect(doors.sendDialogue).toHaveBeenCalledWith({
      message: 'status?',
      sessionId: 'session-a',
      targetProfile: 'review'
    })
    expect(doors.sendVoice).toHaveBeenCalledWith({
      audioId: undefined,
      sessionId: 'session-a',
      targetProfile: undefined,
      transcript: 'approve it'
    })
    expect(doors.respondApproval).toHaveBeenCalledWith({
      actionId: 'approval-a',
      approved: true,
      sessionId: 'session-a'
    })
  })
})
