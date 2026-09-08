import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGroupGateway, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions } from './group-test-utils'
import type { GroupMember } from './types'

// #93813: external writes to a member's per-group session (CLI
// `hermes -p <bot> chat -c "Group: <room>"`, cron, agent tools) must be
// mirrored into the room log instead of silently diverging from it. The
// reconcile sweep runs inside the turn path, right after the pre-turn
// `session.resume` and before the reply-window baseline is taken.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

const LOCAL_MEMBER: GroupMember = { name: 'research', title: '' }

type ChatModule = Record<string, any>
type TurnsModule = Record<string, any>

interface Ctx {
  chat: ChatModule
  gateway: ReturnType<typeof createGroupGateway>
  turns: TurnsModule
  storage: Map<string, unknown>
}

async function loadRoom(options: GatewayOptions = {}): Promise<Ctx> {
  vi.resetModules()
  const gateway = createGroupGateway({ turn: () => 'legacy ok', ...options })

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, turns, shared] = await Promise.all([
    import('./group-chat'),
    import('./group-turns'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, gateway, turns, storage: gateway.storage }
}

/** Seed a room + the member's hidden session holding `rows`, and return the
 *  scripted session so a test can push transcript rows into it. */
async function seedSession(ctx: Ctx, rows: Array<{ content: string; role: string }>) {
  ctx.chat.updateGroupChat('Room', (current: any) => current)
  await ctx.turns.ensureGroupChatSession('Room', LOCAL_MEMBER)

  const sid = String(ctx.chat.$groupChats.get().Room?.sessions?.research)
  const session = ctx.gateway.sessions.get(sid)

  expect(session).toBeTruthy()

  session!.messages.push(...rows)

  return session!
}

const room = (ctx: Ctx) => ctx.chat.$groupChats.get().Room
const texts = (ctx: Ctx) => (room(ctx)?.log || []).map((e: { text: string }) => e.text)
const durableRoom = (ctx: Ctx) => ((ctx.storage.get('group-chats') || {}) as Record<string, any>).Room

describe('external write reconciliation (#93813)', () => {
  beforeEach(() => {
    runTimersInline()
  })

  it('mirrors an external CLI post into the room log before the turn baseline', async () => {
    const ctx = await loadRoom()

    await seedSession(ctx, [
      { content: '[Group chat: "Room"] You are @research. New messages: hi', role: 'user' },
      { content: 'room answer', role: 'assistant' },
      { content: 'external question from the CLI', role: 'user' },
      { content: 'external answer from the CLI', role: 'assistant' }
    ])

    const logLenBefore = texts(ctx).length
    const reply = await ctx.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'next prompt', 't1', [])

    expect(reply).toBe('legacy ok')

    const entryTexts = texts(ctx).slice(logLenBefore)

    // The external pair was mirrored, authored by the member, and nothing else.
    expect(entryTexts.filter((t: string) => t.startsWith('external'))).toEqual([
      'external question from the CLI',
      'external answer from the CLI'
    ])

    for (const entry of room(ctx)!.log.slice(logLenBefore)) {
      if (entry.text.startsWith('external')) {
        expect(entry.from).toMatchObject({ kind: 'member', name: 'research' })
      }
    }

    // The cursor advanced past the last mirrored row so nothing is rescanned.
    expect(durableRoom(ctx)?.externalCursors?.research).toBe(4)
  })

  it('does not mirror room-fed prompts or their replies', async () => {
    const ctx = await loadRoom()

    await seedSession(ctx, [
      { content: '[Group chat: "Room"] You are @research. New messages: hello', role: 'user' },
      { content: 'room-driven reply', role: 'assistant' }
    ])

    const logLenBefore = texts(ctx).length
    await ctx.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'next', 't1', [])

    const entryTexts = texts(ctx).slice(logLenBefore)

    expect(entryTexts).not.toContain('room-driven reply')
    expect(entryTexts.every((t: string) => !t.startsWith('[Group chat:'))).toBe(true)
  })

  it('caps the sweep and advances the cursor only to the last kept row', async () => {
    const ctx = await loadRoom()

    const rows: Array<{ content: string; role: string }> = []

    for (let i = 1; i <= 14; i += 1) {
      rows.push({ content: `external post ${i}`, role: 'user' })
    }

    await seedSession(ctx, rows)

    const logLenBefore = texts(ctx).length
    await ctx.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'next', 't1', [])

    const entryTexts = texts(ctx).slice(logLenBefore).filter((t: string) => /^external post \d+$/.test(t))

    // Only the first 10 were mirrored this turn...
    expect(entryTexts).toEqual(Array.from({ length: 10 }, (_, i) => `external post ${i + 1}`))

    // ...and the cursor sits after post 10, so the next sweep picks up 11-14
    // instead of dropping them.
    expect(durableRoom(ctx)?.externalCursors?.research).toBe(10)
  })

  it('still completes the turn when reconciliation throws', async () => {
    const ctx = await loadRoom()

    await seedSession(ctx, [
      { content: 'external post from a cron job', role: 'user' },
      { content: 'its reply', role: 'assistant' }
    ])

    // Make the room's externalCursors read explode inside the sweep; the
    // turn must proceed as if reconciliation were best-effort (it is
    // wrapped in a try/catch at the call site).
    const roomRecord = room(ctx) as Record<string, any>

    Object.defineProperty(roomRecord, 'externalCursors', {
      get() {
        throw new TypeError('cursor map unavailable')
      },
      configurable: true
    })

    const reply = await ctx.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'still works', 't1', [])

    expect(reply).toBe('legacy ok')
  })
})
