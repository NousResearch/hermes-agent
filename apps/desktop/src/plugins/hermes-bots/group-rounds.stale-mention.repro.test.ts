import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type * as groupTurns from './group-turns'
import type { GroupMember } from './types'

// Reproduction for the "member never answers, message goes stale" report.
//
// Source path proven before writing this (group-rounds.ts):
//   - a failed member turn is swallowed to `reply = null` (catch, ~L665),
//   - the loop STILL advances that member's watermark to log.length (~L706),
//   - the only re-drive for a silent-but-owed member,
//     `unaddressedGroupMentions`, counts ONLY member->member handoffs
//     (`entry.from.kind !== 'member'` is skipped, ~L361).
//
// So when the USER directly @-mentions a member and that member's turn fails
// once on a transient reap, the user's message is consumed (watermark past it)
// and never re-driven — the bot stays silent forever. This test pins the
// CORRECT behavior (the member eventually answers), so it is RED on today's
// code and GREEN once the re-drive covers a user-cited member.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  chat: typeof groupChat
  gateway: ScriptedGateway
  rounds: typeof groupRounds
  turns: typeof groupTurns
}

async function loadRoom(options: GatewayOptions = {}): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway(options)

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, rounds, turns, shared] = await Promise.all([
    import('./group-chat'),
    import('./group-rounds'),
    import('./group-turns'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, gateway, rounds, turns }
}

const MEMBERS: GroupMember[] = [
  { name: 'research', title: '' },
  { name: 'builder', title: '' }
]

const log = (room: Room, group: string) => room.chat.$groupChats.get()[group]?.log || []

async function settle(room: Room, group: string) {
  await drain(() => Boolean(room.chat.$groupChats.get()[group]?.running))
}

beforeEach(() => {
  runTimersInline()
})

describe('user-mentioned member whose first turn fails transiently', () => {
  it('still answers the user (the mention is not permanently consumed)', async () => {
    // builder throws a non-4001 error on its FIRST submit (a transient reap
    // that the turn-level 4001 retry does not cover), then would answer.
    let builderAttempts = 0

    const room = await loadRoom({
      turn: ({ profile }) => {
        if (profile === 'builder') {
          builderAttempts += 1

          if (builderAttempts === 1) {
            throw new Error('transient reap: gateway socket dropped mid-turn')
          }

          return 'builder here — on it.'
        }

        return '(pass)'
      }
    })

    // The user addresses builder directly.
    room.rounds.sendToGroupChat('Council', MEMBERS, '@builder can you take a look?')
    await settle(room, 'Council')

    const memberLines = log(room, 'Council')
      .filter(entry => entry.from.kind === 'member')
      .map(entry => `${entry.from.name}: ${entry.text}`)

    // CORRECT behavior: builder eventually posts its reply. On today's code the
    // failed first turn is swallowed, the watermark is advanced past the user
    // message, and the user-cited member is never re-driven — so builder stays
    // silent and this assertion fails (the bug).
    expect(memberLines.some(line => line.startsWith('builder:'))).toBe(true)
  })
})
