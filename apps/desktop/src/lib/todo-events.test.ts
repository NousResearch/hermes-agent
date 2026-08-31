import { describe, expect, it } from 'vitest'

import { todoSnapshotFromGatewayPayload } from './todo-events'

const todos = [{ content: 'Task', id: 't1', status: 'in_progress' as const }]

describe('todoSnapshotFromGatewayPayload', () => {
  it('adopts a dedicated todo.updated payload', () => {
    expect(todoSnapshotFromGatewayPayload({ generation: 3, revision: 2, session_id: 's1', todos }, 's1')).toEqual({
      generation: 3,
      revision: 2,
      session_id: 's1',
      todos
    })
  })

  it('adds the routed session id to a tool.complete payload', () => {
    expect(todoSnapshotFromGatewayPayload({ generation: 3, revision: 2, todos }, 'runtime-1')).toEqual({
      generation: 3,
      revision: 2,
      session_id: 'runtime-1',
      todos
    })
  })

  it('never trusts a conflicting session id inside an event payload', () => {
    expect(
      todoSnapshotFromGatewayPayload({ generation: 3, revision: 2, session_id: 'other', todos }, 'runtime-1')
    ).toMatchObject({ session_id: 'runtime-1' })
  })

  it('reads the authoritative snapshot from a tool result', () => {
    expect(todoSnapshotFromGatewayPayload({ result: { generation: 4, revision: 4, todos } }, 'runtime-1')).toEqual({
      generation: 4,
      revision: 4,
      session_id: 'runtime-1',
      todos
    })
  })

  it('does not promote an unversioned legacy list to authority', () => {
    expect(todoSnapshotFromGatewayPayload({ todos }, 'runtime-1')).toBeNull()
  })
})
