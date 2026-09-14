import { describe, expect, it } from 'vitest'

import {
  durableGroupChatRooms,
  groupChatSyncSnapshot,
  mergeGroupChatSyncSnapshots,
  mergeRemoteGroupChatSnapshotIntoRooms,
  setGroupChatDescription
} from './group-chat'
import type { GroupChat } from './types'

describe('group chat description', () => {
  it('includes description in groupChatSyncSnapshot', () => {
    const room: GroupChat = {
      description: 'Incident triage and security ops',
      log: [{ at: 1000, from: { kind: 'user', name: 'You' }, text: 'status' }],
      members: [{ name: 'secops' }],
      roomId: 'room-123',
      syncRevision: 1,
      watermarks: {}
    }

    const snapshot = groupChatSyncSnapshot({ 'secops-room': room })
    const compact = snapshot.rooms['id:room-123']
    expect(compact).toBeDefined()
    expect(compact.description).toBe('Incident triage and security ops')
  })

  it('merges descriptions in mergeGroupChatSyncSnapshots favoring higher revision', () => {
    const local = {
      version: 3 as const,
      updatedAt: 1000,
      rooms: {
        'id:room-1': {
          name: 'ops',
          roomId: 'room-1',
          description: 'Local older description',
          revision: 1,
          log: [],
          members: []
        }
      }
    }

    const remote = {
      version: 3 as const,
      updatedAt: 2000,
      rooms: {
        'id:room-1': {
          name: 'ops',
          roomId: 'room-1',
          description: 'Remote updated description',
          revision: 2,
          log: [],
          members: []
        }
      }
    }

    const merged = mergeGroupChatSyncSnapshots(local, remote)
    expect(merged.rooms['id:room-1']?.description).toBe('Remote updated description')
  })

  it('carries description in mergeRemoteGroupChatSnapshotIntoRooms', () => {
    const remote = {
      version: 3 as const,
      updatedAt: 2000,
      rooms: {
        'id:room-1': {
          name: 'secops',
          roomId: 'room-1',
          description: 'Automated vulnerability triage',
          revision: 5,
          log: [],
          members: []
        }
      }
    }

    const current: Record<string, GroupChat> = {
      secops: {
        roomId: 'room-1',
        log: [],
        syncRevision: 1,
        watermarks: {}
      }
    }

    const result = mergeRemoteGroupChatSnapshotIntoRooms(remote, current)
    expect(result.secops.description).toBe('Automated vulnerability triage')
  })

  it('preserves description in durableGroupChatRooms', () => {
    const all: Record<string, GroupChat> = {
      devs: {
        description: 'Core engineering sync',
        log: [],
        roomId: 'room-devs',
        syncRevision: 1,
        watermarks: {}
      }
    }

    const durable = durableGroupChatRooms(all)
    expect(durable.devs.description).toBe('Core engineering sync')
  })

  it('setGroupChatDescription updates description and trims whitespace', () => {
    setGroupChatDescription('test-room', '  Helpful bot cluster  ')
    const durable = durableGroupChatRooms()
    expect(durable['test-room']?.description).toBe('Helpful bot cluster')

    setGroupChatDescription('test-room', '   ')
    const cleared = durableGroupChatRooms()
    expect(cleared['test-room']?.description).toBeUndefined()
  })

  it('setGroupChatDescription clamps description to 512 characters', () => {
    const longDesc = 'a'.repeat(600)
    setGroupChatDescription('clamped-room', longDesc)
    const durable = durableGroupChatRooms()
    expect(durable['clamped-room']?.description).toHaveLength(512)
    expect(durable['clamped-room']?.description).toBe('a'.repeat(512))
  })
})
