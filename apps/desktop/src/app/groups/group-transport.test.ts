import { describe, expect, it, vi } from 'vitest'

import { createGroupTransport } from './group-transport'

describe('group transport', () => {
  it('uses the group RPC contract for lifecycle and messages', async () => {
    const request = vi.fn(async (_method: string, _params?: Record<string, unknown>): Promise<unknown> => ({ ok: true }))
    const transport = createGroupTransport(request)

    await transport.listRooms()
    await transport.createRoom({ name: 'Ship', profiles: ['a', 'b'], workspace: '/work/ship' })
    await transport.getRoom('r1')
    await transport.getRoom('r1', { beforeSeq: 41, cursor: 'older-1' })
    await transport.sendMessage('r1', '@all go', ['all'])
    await transport.interrupt('r1')
    await transport.subscribe('r1')
    await transport.unsubscribe('r1')
    await transport.deleteRoom('r1')

    const methods = request.mock.calls.map(call => (call as unknown as [string])[0])
    expect(methods).toEqual([
      'group.room.list', 'group.room.create', 'group.room.get', 'group.room.get', 'group.message.send',
      'group.run.interrupt', 'group.subscribe', 'group.unsubscribe', 'group.room.delete'
    ])
    expect(request).toHaveBeenCalledWith('group.message.send', {
      room_id: 'r1', content: '@all go', mentions: ['all']
    })
    expect(request).toHaveBeenCalledWith('group.room.create', {
      name: 'Ship', profiles: ['a', 'b'], workspace: '/work/ship'
    })
    expect(request).toHaveBeenCalledWith('group.room.get', {
      room_id: 'r1', before_seq: 41, cursor: 'older-1'
    })
  })
})
