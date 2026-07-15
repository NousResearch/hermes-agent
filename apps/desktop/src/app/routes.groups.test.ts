import { describe, expect, it } from 'vitest'

import { appViewForPath, groupRoomId, groupRoomRoute } from './routes'

describe('group routes', () => {
  it('recognizes the groups index and durable room route', () => {
    expect(appViewForPath('/groups')).toBe('groups')
    expect(appViewForPath('/groups/room%201')).toBe('groups')
    expect(groupRoomId('/groups/room%201')).toBe('room 1')
    expect(groupRoomRoute('room 1')).toBe('/groups/room%201')
  })
})
