import { beforeEach, describe, expect, it } from 'vitest'

import type { GroupRoom } from './group-model'
import {
  $groupState,
  beginGroupRoomsRequest,
  cacheGroupRoom,
  cacheSentGroupMessage,
  clearCachedGroupApproval,
  reconcileGroupRooms
} from './group-store'

const room = (id: string): GroupRoom => ({ id, name: id, profiles: [], messages: [], runningProfiles: [] })

describe('group store authoritative cache', () => {
  beforeEach(() => $groupState.set({ rooms: {} }))

  it('removes rooms absent from the latest authoritative list', () => {
    cacheGroupRoom(room('old'))
    const request = beginGroupRoomsRequest()

    reconcileGroupRooms(request, [room('new')])

    expect(Object.keys($groupState.get().rooms)).toEqual(['new'])
  })

  it('ignores stale list responses that arrive after a newer request', () => {
    const stale = beginGroupRoomsRequest()
    const latest = beginGroupRoomsRequest()

    reconcileGroupRooms(latest, [room('new')])
    reconcileGroupRooms(stale, [room('old')])

    expect(Object.keys($groupState.get().rooms)).toEqual(['new'])
  })

  it('does not let a list started before a live room update erase that update', () => {
    const stale = beginGroupRoomsRequest()
    cacheGroupRoom(room('live'))

    reconcileGroupRooms(stale, [])

    expect(Object.keys($groupState.get().rooms)).toEqual(['live'])
  })

  it('immediately caches the backend user.message returned by send', () => {
    cacheGroupRoom(room('r1'))

    cacheSentGroupMessage({
      event: { room_id: 'r1', seq: 7, type: 'user.message', created_at: 123, payload: { text: 'Ship it' } }
    })

    expect($groupState.get().rooms.r1.messages).toEqual([
      expect.objectContaining({ id: 'group-7', role: 'user', content: 'Ship it', status: 'complete' })
    ])
  })

  it('clears approval state only after a successful response', () => {
    cacheGroupRoom({
      ...room('r1'),
      messages: [{
        id: 'm1', role: 'assistant', content: '', parts: [], status: 'approval', runtimeSessionId: 's1',
        approval: { choices: ['once', 'deny'], command: 'rm x', description: 'Delete' }
      }]
    })

    clearCachedGroupApproval('r1', 'm1')

    expect($groupState.get().rooms.r1.messages[0]).toEqual(expect.objectContaining({ status: 'streaming' }))
    expect($groupState.get().rooms.r1.messages[0].approval).toBeUndefined()
  })
})
