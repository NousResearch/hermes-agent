import { afterEach, describe, expect, it, vi } from 'vitest'

import { $groupChats } from './group-chat'
import {
  bindGroupLaunchBridge,
  GROUP_FEED_REQUEST_EVENT,
  GROUP_OPEN_EVENT,
  GROUP_POST_EVENT,
  groupFeed,
  groupLaunchOptions,
  GROUPS_CHANGED_EVENT,
  GROUPS_REQUEST_EVENT
} from './group-launch-bridge'
import type { GroupChat } from './types'

const room = (overrides: Partial<GroupChat> = {}): GroupChat =>
  ({ log: [], watermarks: {}, members: [], ...overrides }) as GroupChat

afterEach(() => {
  $groupChats.set({})
})

describe('groupLaunchOptions', () => {
  it('lists live rooms alphabetically with their member counts, skipping tombstones', () => {
    const options = groupLaunchOptions({
      Zebra: room({ members: [{ name: 'a' }, { name: 'b' }] as GroupChat['members'] }),
      Alpha: room(),
      Gone: room({ tombstone: true }),
      Broken: { members: [] } as unknown as GroupChat
    })

    expect(options).toEqual([
      { displayName: 'Alpha', groupId: 'Alpha', memberCount: 0, reachable: true },
      { displayName: 'Zebra', groupId: 'Zebra', memberCount: 2, reachable: true }
    ])
  })
})

describe('bindGroupLaunchBridge', () => {
  it('answers a groups request from the room store', () => {
    $groupChats.set({ Design: room() })
    const dispose = bindGroupLaunchBridge(vi.fn())

    let answer: unknown = null
    window.dispatchEvent(new CustomEvent(GROUPS_REQUEST_EVENT, { detail: { respond: (o: unknown) => (answer = o) } }))

    expect(answer).toEqual([{ displayName: 'Design', groupId: 'Design', memberCount: 0, reachable: true }])
    dispose()
  })

  it('opens a live room and refuses an unknown or tombstoned one', () => {
    $groupChats.set({ Design: room(), Gone: room({ tombstone: true }) })
    const open = vi.fn()
    const dispose = bindGroupLaunchBridge(open)

    const ask = (groupId: string) => {
      let ok: boolean | null = null
      window.dispatchEvent(new CustomEvent(GROUP_OPEN_EVENT, { detail: { groupId, respond: (v: boolean) => (ok = v) } }))

      return ok
    }

    expect(ask('Design')).toBe(true)
    expect(open).toHaveBeenCalledWith('Design')
    expect(ask('Gone')).toBe(false)
    expect(ask('Nope')).toBe(false)
    expect(open).toHaveBeenCalledTimes(1)
    dispose()
  })

  it('announces room changes and stops after dispose', () => {
    const changed = vi.fn()
    window.addEventListener(GROUPS_CHANGED_EVENT, changed)
    const dispose = bindGroupLaunchBridge(vi.fn())

    $groupChats.set({ New: room() })
    expect(changed).toHaveBeenCalledTimes(1)

    dispose()
    $groupChats.set({})
    expect(changed).toHaveBeenCalledTimes(1)
    window.removeEventListener(GROUPS_CHANGED_EVENT, changed)
  })
})

describe('room feed + post for the launchers', () => {
  it('groupFeed maps the tail of the log and the running member', () => {
    const log = Array.from({ length: 45 }, (_, i) => ({
      at: i,
      from: i % 2 ? { kind: 'member' as const, name: 'gary' } : { kind: 'user' as const, name: 'You' },
      text: `m${i}`
    }))

    const feed = groupFeed('Design', {
      Design: { ...room({ members: [{ name: 'gary' }] as GroupChat['members'] }), log, running: true, turn: 'gary' } as GroupChat
    })

    expect(feed?.entries).toHaveLength(40)
    expect(feed?.entries[0]?.text).toBe('m5')
    expect(feed?.entries[39]).toMatchObject({ author: 'You', kind: 'user', text: 'm44' })
    expect(feed?.members).toEqual(['gary'])
    expect(feed?.turn).toBe('gary')
    expect(feed?.running).toBe(true)
    expect(groupFeed('Nope')).toBeNull()
  })

  it('answers a feed request and refuses a post to an unknown room', () => {
    $groupChats.set({ Design: room() })
    const dispose = bindGroupLaunchBridge(vi.fn())

    let answer: unknown = 'unset'
    window.dispatchEvent(
      new CustomEvent(GROUP_FEED_REQUEST_EVENT, { detail: { groupId: 'Design', respond: (f: unknown) => (answer = f) } })
    )
    expect(answer).toMatchObject({ groupId: 'Design', entries: [] })

    let ok: boolean | null = null
    window.dispatchEvent(
      new CustomEvent(GROUP_POST_EVENT, { detail: { groupId: 'Nope', text: 'hi', respond: (v: boolean) => (ok = v) } })
    )
    expect(ok).toBe(false)
    dispose()
  })
})
