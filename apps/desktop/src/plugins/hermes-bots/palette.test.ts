import { beforeEach, describe, expect, it, vi } from 'vitest'

const { roster, meta, openRosterBot } = vi.hoisted(() => ({
  roster: { current: [] as Array<{ name: string; connectionId: string; hidden?: boolean }> },
  meta: { current: {} },
  openRosterBot: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', () => ({ PALETTE_AREA: 'palette' }))
vi.mock('./data', async () => {
  const { atom } = await import('nanostores')

  return {
    $botMeta: atom(meta.current),
    $lastRoster: atom([]),
    botRosterKey: (bot: { name: string; connectionId: string }) => `${bot.connectionId}::${bot.name}`,
    cachedUnionRoster: () => ({ profiles: roster.current })
  }
})
vi.mock('./hidden-bots', () => ({ isBotHidden: (bot: { hidden?: boolean }) => Boolean(bot.hidden) }))
vi.mock('./labels', () => ({ displayName: (bot: { name: string }) => bot.name }))
vi.mock('./routing', () => ({ botRosterMeta: () => null }))
vi.mock('./roster-actions', () => ({ openRosterBot }))

import { registerBotChatPalette } from './palette'

describe('Bot Chat command palette', () => {
  beforeEach(() => {
    roster.current = []
    openRosterBot.mockClear()
  })

  it('opens the selected profile through the canonical roster action', () => {
    const rows = [
      { name: 'ops', connectionId: 'home' },
      { name: 'ops', connectionId: 'other' },
      { name: 'private', connectionId: 'home', hidden: true }
    ]

    roster.current = rows
    const items = new Map<string, { data: { label: string; run: () => void } }>()
    const dispose: Array<() => void> = []

    const ctx = {
      register: (item: { id: string; data: { label: string; run: () => void } }) => {
        items.set(item.id, item)

        return () => void items.delete(item.id)
      },
      onDispose: (fn: () => void) => dispose.push(fn)
    }

    const sync = registerBotChatPalette(ctx as never)
    expect([...items.values()].map(item => item.data.label)).toEqual(['Chat with ops (home)', 'Chat with ops (other)'])

    const run = items.get('chat-other::ops')!.data.run
    run()
    expect(openRosterBot).toHaveBeenCalledWith(rows[1])

    roster.current = [rows[0]]
    sync()
    expect(items.has('chat-other::ops')).toBe(false)
    dispose.forEach(fn => fn())
    expect(items.size).toBe(0)
  })
})
