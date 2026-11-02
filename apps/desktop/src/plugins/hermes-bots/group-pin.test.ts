/**
 * "Pin to top" for group chats (hermes-agent#89813): the flag the roster sort
 * already honours gets a write site on the room record, and the pin survives
 * a reload through the same durable group-chats persistence the room uses —
 * never through any bot's profile meta.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const { storage } = vi.hoisted(() => ({ storage: new Map<string, unknown>() }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return { atom, host: {} }
})

vi.mock('./data', async () => {
  const { atom } = await import('nanostores')

  return { $botMeta: atom({}), $lastRoster: atom([]), botRosterKey: (bot: { name?: string }) => `legacy::${bot?.name}` }
})

vi.mock('./routing', () => ({ botRosterMeta: () => undefined }))

vi.mock('./shared', () => ({
  getPluginCtx: () => ({
    storage: {
      get: (key: string, fallback: unknown) => (storage.has(key) ? storage.get(key) : fallback),
      set: (key: string, value: unknown) => storage.set(key, value)
    }
  })
}))

import { $groupChats } from './group-chat'
import { sortGroupRosterRows } from './group-order'
import { toggleGroupChatPinned } from './group-pin'

beforeEach(() => {
  storage.clear()
  $groupChats.set({ Standup: { log: [], watermarks: {}, sessions: {}, stranded: {}, members: [] } as never })
})

describe('group chat pin', () => {
  it('pins the room record durably and floats the room above unpinned rows; a second toggle unpins', () => {
    expect(toggleGroupChatPinned('Standup')).toBe(true)
    expect($groupChats.get().Standup.pinned).toBe(true)
    expect((storage.get('group-chats') as Record<string, { pinned?: boolean }>).Standup.pinned).toBe(true)

    const rows = [
      { kind: 'bot' as const, name: 'busy-bot', pinned: false, activity: 900 },
      { kind: 'group' as const, name: 'Standup', pinned: Boolean($groupChats.get().Standup.pinned), activity: 100 }
    ]

    expect(sortGroupRosterRows(rows, $groupChats.get()).map(row => row.name)).toEqual(['Standup', 'busy-bot'])

    expect(toggleGroupChatPinned('Standup')).toBe(false)
    expect((storage.get('group-chats') as Record<string, { pinned?: boolean }>).Standup.pinned).toBe(false)
    // A room that no longer exists is not resurrected by a stale menu.
    expect(toggleGroupChatPinned('Gone')).toBeNull()
    expect($groupChats.get().Gone).toBeUndefined()
  })
})
