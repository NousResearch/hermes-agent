import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type * as groupTurns from './group-turns'
import type { Attachment, GroupChat, GroupMember } from './types'

// One member's turn: resolve its per-room session, stage attachments, submit,
// and poll until the gateway says the member is done. The session is the
// fragile part — it is addressed by a STORED id that outlives every runtime
// id, and the socket it rides can be reaped underneath a turn in flight.

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

const LOCAL_MEMBER: GroupMember = { name: 'helper', title: '' }
const ROUTED_MEMBER: GroupMember = { connectionId: 'mini', name: 'helper', remoteSource: true }
const IMG: Attachment = { data: 'data:image/png;base64,iVBORw0KGgo=', kind: 'image', name: 'shot.png' }

const log = (room: Room, group: string) => room.chat.$groupChats.get()[group]?.log || []

beforeEach(() => {
  runTimersInline()
})

describe('session resolution', () => {
  it('pins session titles to the roomId, with a legacy fallback to the display name', async () => {
    const room = await loadRoom()

    // Rooms persisted before roomIds keep name-based titles so their existing
    // \"Group: <name>\" sessions keep resolving after an upgrade.
    room.chat.updateGroupChat('Legacy', current => current)
    const legacy = await room.turns.ensureGroupChatSession('Legacy', { name: 'research', title: '' })

    expect(room.gateway.sessions.get(String(legacy.stored))?.title).toBe('Group: Legacy')

    // New rooms pin the title to the immutable roomId, never the display name.
    room.chat.updateGroupChat('New', current => {
      current.roomId = 'r-abc'

      return current
    })
    const fresh = await room.turns.ensureGroupChatSession('New', { name: 'research', title: '' })

    expect(room.gateway.sessions.get(String(fresh.stored))?.title).toBe('Group: r-abc')
  })

  it('creates member sessions with the room_plumbing + follow_profile_config contracts', async () => {
    // The PR #97008 contracts: room member sessions always rebuild from the
    // member profile's CURRENT config on resume, never a stale stored
    // model/provider pin. Dropping either param silently regresses rooms to
    // the server's hidden + \"Group: \" title legacy fallback.
    const room = await loadRoom()

    room.chat.updateGroupChat('Contract', current => {
      current.roomId = 'r-contract'

      return current
    })
    const handle = await room.turns.ensureGroupChatSession('Contract', { name: 'research', title: '' })

    expect(room.gateway.sessions.get(String(handle.stored))?.contracts).toEqual({
      follow_profile_config: true,
      room_plumbing: true
    })
  })

  it('mints fresh member sessions when a same-name group is recreated after disband', async () => {
    const room = await loadRoom()
    const member: GroupMember = { name: 'research', title: '' }

    room.chat.updateGroupChat('Alpha', current => {
      current.roomId = 'r-one'

      return current
    })
    const first = await room.turns.ensureGroupChatSession('Alpha', member)

    // Disband: the room record is gone; the member's gateway session survives.
    const rooms = { ...room.chat.$groupChats.get() }
    delete rooms.Alpha
    room.chat.$groupChats.set(rooms)

    // Recreate under the same display name with a freshly minted roomId.
    room.chat.updateGroupChat('Alpha', current => {
      current.roomId = 'r-two'

      return current
    })
    const second = await room.turns.ensureGroupChatSession('Alpha', member)

    expect(second.stored).not.toBe(first.stored)
    expect(room.gateway.sessions.get(String(first.stored))?.title).toBe('Group: r-one')
    expect(room.gateway.sessions.get(String(second.stored))?.title).toBe('Group: r-two')
  })

  it('fails closed on a transient resume failure instead of forking the member session', async () => {
    const room = await loadRoom()
    const member: GroupMember = { name: 'research', title: '' }

    room.chat.updateGroupChat('Core', current => {
      current.roomId = 'r-core'

      return current
    })
    await room.turns.ensureGroupChatSession('Core', member)

    // Record the stored id a legitimate resume would target.\n    const stored = room.chat.$groupChats.get().Core?.sessions?.research || null

    expect(stored).toBeTruthy()

    // Simulate a transient resume failure: the gateway rejects without 4007.\n    room.gateway.host.requestProfile = vi.fn(async (_, method: string) => {
      if (method === 'session.resume') {
        throw new Error('ECONNRESET')
      }

      return {}
    })

    await expect(room.turns.ensureGroupChatSession('Core', member)).rejects.toThrow(/not starting a new one/)
    // The session store was never updated — the old sid remains, so the next\n    // successful resume can still recover it.\n    expect(room.chat.$groupChats.get().Core?.sessions?.research).toBe(stored)
  })

  it('falls through a genuine 4007 to the title lookup, then creates', async () => {
    const room = await loadRoom()

    // The session record is orphaned: the stored sid is dead but the room\n    // still references it. A real resume failure (4007) falls through the\n    // stored-id path and tries the title-based lookup, which recovers from\n    // a session that is renamed or otherwise disconnected from its stored id.\n    room.chat.updateGroupChat('Core', current => {
      current.roomId = 'r-core'
      current.sessions = { research: 'sid-orphaned' }

      return current
    })
    const handle = await room.turns.ensureGroupChatSession('Core', { name: 'research', title: '' })

    expect(handle.runtime).toBeTruthy()
    expect(room.gateway.sessions.get(String(handle.stored))?.title).toBe('Group: r-core')
  })
})

describe('session-gone classification', () => {
  it('treats 4001 and \"not in memory\" as recoverable, 4007 as not', async () => {
    const { turns } = await loadRoom()

    expect(turns.isSessionGoneError({ code: 4001 })).toBe(true)
    expect(turns.isSessionGoneError({ code: 4007 })).toBe(false)
    expect(turns.isSessionGoneError({ message: 'session not found: sid-x' })).toBe(true)
    expect(turns.isSessionGoneError({ message: 'RPC rejected: not in memory' })).toBe(true)
  })

  it('recovers a 4001 on the first submit via the STORED id and delivers', async () => {
    const room = await loadRoom({
      failFirstSubmitWith: { code: 4001, message: 'not in memory' },
      turn: () => 'recovery ok'
    })

    room.chat.updateGroupChat('Core', current => current)
    await room.turns.ensureGroupChatSession('Core', { name: 'research', title: '' })

    const reply = await room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'hi there', 't1', [])

    expect(reply).toBe('recovery ok')
  })

  it('does not retry a persistent non-4001 submit failure', async () => {
    const room = await loadRoom({
      failEverySubmitWith: { code: 7991, message: 'model unavailable' },
      turn: () => 'never runs'
    })

    await expect(room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'hi', 't1', [])).rejects.toThrow(
      'model unavailable'
    )
  })
})

describe('per-turn socket lease', () => {
  it('is acquired before any session RPC and held across attach + submit', async () => {
    const room = await loadRoom({ turn: () => 'ack' })

    room.chat.updateGroupChat('Room', current => current)
    await room.turns.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'look at this', 't1', [IMG])

    const firstRequestIndex = room.gateway.timeline.findIndex(entry => entry.startsWith('session'))

    expect(room.gateway.timeline[firstRequestIndex - 1]).toBe('retain')
    expect(room.gateway.refcount()).toBe(0)
  })

  it('is released after the turn — the refcount returns to zero', async () => {
    const room = await loadRoom({ turn: () => 'ok' })

    await room.turns.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'hi', 't1', [])

    expect(room.gateway.refcount()).toBe(0)
  })

  it('is released even when the turn fails', async () => {
    const room = await loadRoom({ failEverySubmitWith: new Error('down') })

    await expect(room.turns.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'hi', 't1', [])).rejects.toThrow()

    expect(room.gateway.refcount()).toBe(0)
  })

  it('is feature-detected: hosts without retainProfile still run the turn', async () => {
    const room = await loadRoom({ turn: () => 'legacy ok' })
    delete room.gateway.host.retainProfile

    expect(await room.turns.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'hi', 't1', [])).toBe('legacy ok')
  })
})

describe('push-woken poll', () => {
  it(\"wakes on the member session's message.complete instead of sleeping out the backstop\", async () => {
    const room = await loadRoom({
      pollsBusy: 10,
      turn: () => 'done'
    })

    // host.onEvent is the only way the poll loop wakes faster than 5 seconds.
    // Without it, the test would observe ~10 post-submit resumes — the poll\n    // looping at `GROUP_TURN_POLL_MS` cadence. With the event tap the turn\n    // wakes on the FIRST terminal frame and exits the poll quickly.\n    const eventSubs: Array<() => void> = []

    // eslint-disable-next-line @typescript-eslint/no-explicit-any\n    room.gateway.host.onEvent = vi.fn((type: string, callback: any) => {
      const unsub = () => undefined

      eventSubs.push(unsub)

      if (type === 'message.complete') {
        callback({ session_id: room.gateway.sessions.values().next().value?.runtime })
      }

      return unsub
    })
    const turn = room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'hi', 't1', [])

    await drain(() => Boolean(room.chat.$groupChats.get().Room?.running))

    const reply = await turn
    const resumeCount = room.gateway.rpcFor('session.resume').length

    expect(reply).toBe('done')
    expect(resumeCount).toBeLessThan(5)
    expect(eventSubs.length).toBeGreaterThan(0)

    for (const unsub of eventSubs) {
      unsub()
    }
  })
})

describe('reply selection (#94376)', () => {
  it('surfaces a substantive answer followed by a synthetic continuation (pass)', async () => {
    const room = await loadRoom({
      turn: () => [
        { content: 'Detailed analysis follows', role: 'assistant' },
        { content: '(pass)', role: 'assistant' }
      ]
    })

    expect(await room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'Did you welcome them?', 't1', [])).toBe(
      'Detailed analysis follows'
    )
  })

  it('still reads a genuine pass-only turn as silent', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    expect(await room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'hi', 't1', [])).toBe(null)
  })
})

describe('clarify and approvals (#90694)', () => {
  const CLARIFY = {
    choices: ['staging', 'prod'],
    multi_select: false,
    question: 'Which env should I target?',
    request_id: 'req-clarify-1'
  }

  const APPROVAL = {
    choices: ['once', 'session', 'deny'],
    command: 'rm -rf ./build',
    description: 'Clean the build directory',
    request_id: 'req-approval-1'
  }

  it('holds the turn open while a member is blocked on clarify, then lands the reply', async () => {
    const room = await loadRoom({
      clarifyUntil: { research: { payload: CLARIFY, until: 3 } },
      turn: () => 'targeting staging'
    })

    const thread = room.rounds.sendToGroupChat(
      'Core',
      [{ name: 'research', title: '' }],
      '@research deploy it',
      null,
      []
    )

    await drain(() => Boolean(room.chat.$groupChats.get().Core?.running))

    const replies = log(room, 'Core').filter(entry => entry.thread === thread && entry.from.kind === 'member')

    expect(replies).toHaveLength(1)
    expect(replies[0].text).toBe('targeting staging')
    expect(Object.keys(room.chat.$groupClarify.get())).toHaveLength(0)
    // The mirror pass ran while the question was blocking, badging the room.
    // A poll that never inspects pending_clarify leaves this unset — it is the
    // observable proof the gate executed.
    expect(room.chat.$groupNeedsYou.get().Core).toBe(true)
  })

  it('mirrors a question, badges needs-you, and is idempotent per request', async () => {
    const { chat, turns } = await loadRoom()
    const member: GroupMember = { name: 'research', title: '' }

    expect(turns.syncGroupClarify('Core', member, { pending_clarify: CLARIFY })).toBe(true)

    const mirrored = Object.values(chat.$groupClarify.get())

    expect(mirrored).toHaveLength(1)
    expect(mirrored[0].requestId).toBe('req-clarify-1')
    expect(mirrored[0].question).toBe('Which env should I target?')
    expect(mirrored[0].choices).toEqual(['staging', 'prod'])
    expect(chat.$groupNeedsYou.get().Core).toBe(true)

    // Same request again: no new entry, identity preserved.
    turns.syncGroupClarify('Core', member, { pending_clarify: CLARIFY })

    expect(Object.values(chat.$groupClarify.get())[0]).toBe(mirrored[0])

    // Question resolved server-side: the mirror clears.
    expect(turns.syncGroupClarify('Core', member, {})).toBe(false)
    expect(Object.keys(chat.$groupClarify.get())).toHaveLength(0)
  })

  it('never mirrors a question for older backends without pending_clarify', async () => {
    const { chat, turns } = await loadRoom()

    expect(turns.syncGroupClarify('Core', { name: 'research' }, { messages: [] })).toBe(false)
    expect(Object.keys(chat.$groupClarify.get())).toHaveLength(0)
  })

  it('routes an answer through clarify.respond and clears the mirror', async () => {
    const room = await loadRoom()
    const member: GroupMember = { name: 'research', title: '' }

    room.turns.syncGroupClarify('Core', member, { pending_clarify: CLARIFY })
    await room.turns.answerGroupClarify(Object.values(room.chat.$groupClarify.get())[0], member, 'staging')

    expect(room.gateway.rpcFor('clarify.respond').map(call => call.params)).toEqual([
      { answer: 'staging', request_id: 'req-clarify-1' }
    ])
    expect(Object.keys(room.chat.$groupClarify.get())).toHaveLength(0)
  })

  it('sends one respond per batch question, in order', async () => {
    const room = await loadRoom()
    const member: GroupMember = { name: 'research', title: '' }

    room.turns.syncGroupClarify('Core', member, {
      pending_clarify: {
        questions: [
          { choices: ['staging', 'prod'], qid: 'q0', question: 'Env?' },
          { choices: [], qid: 'q1', question: 'Region?' }
        ],
        request_id: 'req-batch-1'
      }
    })
    await room.turns.answerGroupClarify(Object.values(room.chat.$groupClarify.get())[0], member, {
      q0: 'staging',
      q1: 'eu-west'
    })

    expect(room.gateway.rpcFor('clarify.respond').map(call => call.params)).toEqual([
      { answer: 'staging', question_id: 'q0', request_id: 'req-batch-1' },
      { answer: 'eu-west', question_id: 'q1', request_id: 'req-batch-1' }
    ])
    expect(Object.keys(room.chat.$groupClarify.get())).toHaveLength(0)
  })

  it('clears only the disbanded room's mirrored questions', async () => {
    const room = await loadRoom()

    room.turns.syncGroupClarify('Core', { name: 'research' }, { pending_clarify: CLARIFY })
    room.turns.syncGroupClarify('Other', { name: 'ops' }, { pending_clarify: { ...CLARIFY, request_id: 'req-2' } })
    room.turns.clearGroupClarify('Core')

    const remaining = Object.values(room.chat.$groupClarify.get())

    expect(remaining).toHaveLength(1)
    expect(remaining[0].group).toBe('Other')
  })

  it('keeps a clarify prompt bound to the room after a mid-poll rename (#101568)', async () => {
    const room = await loadRoom({
      clarifyUntil: { research: { payload: CLARIFY, until: 3 } },
      onResumePoll: () => {
        room.chat.renameGroupChat('OldName', 'NewName')
      },
      turn: () => 'targeting staging'
    })

    room.chat.updateGroupChat('OldName', current => {
      current.roomId = 'r-oldname'

      return current
    })
    const thread = room.rounds.sendToGroupChat(
      'OldName',
      [{ name: 'research', title: '' }],
      '@research deploy it',
      null,
      []
    )

    await drain(() => Boolean(room.chat.$groupChats.get().NewName?.running))

    const replies = log(room, 'NewName').filter(entry => entry.thread === thread && entry.from.kind === 'member')

    expect(replies).toHaveLength(1)
    expect(replies[0].text).toBe('targeting staging')
    // The clarify mirror is gone, never orphaned under OldName.
    expect(Object.keys(room.chat.$groupClarify.get())).toHaveLength(0)
    expect(room.chat.$groupNeedsYou.get().NewName).toBe(true)
  })

  it('keeps an approval prompt bound to the room after a mid-poll rename (#101568)', async () => {
    const room = await loadRoom({
      approvalUntil: { research: { payload: APPROVAL, until: 2 } },
      onResumePoll: () => {
        room.chat.renameGroupChat('Before', 'After')
      },
      turn: () => 'approved once'
    })

    room.chat.updateGroupChat('Before', current => {
      current.roomId = 'r-before'

      return current
    })
    const thread = room.rounds.sendToGroupChat(
      'Before',
      [{ name: 'research', title: '' }],
      '@research cleanup',
      null,
      []
    )

    await drain(() => Boolean(room.chat.$groupChats.get().After?.running))

    const replies = log(room, 'After').filter(entry => entry.thread === thread && entry.from.kind === 'member')

    expect(replies).toHaveLength(1)
    expect(replies[0].text).toBe('approved once')
    expect(room.chat.$groupNeedsYou.get().After).toBe(true)
  })

  it('holds the turn open on a command approval too', async () => {
    const room = await loadRoom({
      approvalUntil: { research: { payload: APPROVAL, until: 2 } },
      turn: () => 'cleaned'
    })

    const thread = room.rounds.sendToGroupChat(
      'Core',
      [{ name: 'research', title: '' }],
      '@research cleanup',
      null,
      []
    )

    await drain(() => Boolean(room.chat.$groupChats.get().Core?.running))

    const replies = log(room, 'Core').filter(entry => entry.thread === thread && entry.from.kind === 'member')

    expect(replies).toHaveLength(1)
    expect(replies[0].text).toBe('cleaned')
    // The mirror pass ran, so the approval request must have shown at least
    // once during the poll.
    expect(room.chat.$groupNeedsYou.get().Core).toBe(true)
  })

  it('mirrors an approval with its kind, command and server choices', async () => {
    const { chat, turns } = await loadRoom()

    expect(turns.syncGroupClarify('Core', { name: 'ops' }, { pending_approval: APPROVAL })).toBe(true)

    const mirrored = Object.values(chat.$groupClarify.get())

    expect(mirrored).toHaveLength(1)
    expect(mirrored[0].kind).toBe('approval')
    expect(mirrored[0].question).toBe('Clean the build directory')
    expect(mirrored[0].command).toBe('rm -rf ./build')
    expect(mirrored[0].choices).toEqual(['once', 'session', 'deny'])
  })

  it('falls back to once/deny when the server sends no choice set', async () => {
    const { chat, turns } = await loadRoom()

    turns.syncGroupClarify('Core', { name: 'ops' }, { pending_approval: { ...APPROVAL, choices: [] } })

    expect(Object.values(chat.$groupClarify.get())[0].choices).toEqual(['once', 'deny'])
  })

  it('routes approvals through approval.respond with the session and choice', async () => {
    const room = await loadRoom()

    room.turns.syncGroupClarify('Core', { name: 'ops' }, { pending_approval: { ...APPROVAL }, session_id: 'sid-1' })

    const prompt = Object.values(room.chat.$groupClarify.get())[0]
    await room.turns.answerGroupClarify(prompt, { name: 'ops' }, 'session')

    expect(room.gateway.rpcFor('approval.respond').map(call => call.params)).toEqual([
      { choice: 'session', request_id: 'req-approval-1', session_id: 'sid-1' }
    ])
  })

  it('lets clarify outrank approval when a snapshot carries both', async () => {
    const { chat, turns } = await loadRoom()

    turns.syncGroupClarify('Core', { name: 'research' }, {
      pending_approval: APPROVAL,
      pending_clarify: CLARIFY
    })

    const mirrored = Object.values(chat.$groupClarify.get())

    expect(mirrored).toHaveLength(1)
    expect(mirrored[0].kind).toBe('clarify')
  })

  // #101568: poll-after-rename race — the poll loop must stay bound to the\n  // room's current identity across rename.
  it('keeps an in-flight clarify bound to the renamed room', async () => {
    const room = await loadRoom({
      clarifyUntil: { research: { payload: CLARIFY, until: 3 } },
      onResumePoll: (polls: number) => {
        // Mid-turn rename: after the first poll ran (it mirrored the clarify\n        // under 'Core'), rename the room to 'Renamed'.\n        if (polls === 1) {
          const all = { ...room.chat.$groupChats.get() }
          const oldRoom = all.Core || {}
          delete all.Core
          all.Renamed = { ...oldRoom }
          room.chat.$groupChats.set(all)

          // clearGroupClarify('Core') simulates what renameGroupChat does.\n          room.turns.clearGroupClarify('Core')

          // The next poll must NOT recreate 'Core::research' — it must write\n          // under 'Renamed::research' instead.
        }
      },
      turn: () => 'answer under renamed room'
    })

    room.chat.updateGroupChat('Core', current => {
      current.roomId = 'r-core'

      return current
    })
    await room.turns.runGroupChatMemberTurn('Core', { name: 'research', title: '' }, 'hi', 't1', [])

    // Poll wrote under the live name, not the retired 'Core'.\n    expect(Object.keys(room.chat.$groupClarify.get())).not.toContain('Core::research')

    // The prompt must still be accessible under 'Renamed'.\n    const remaining = Object.values(room.chat.$groupClarify.get())
    expect(remaining.filter(p => p.group === 'Renamed')).toHaveLength(0) // cleared after answer
  })

  it('keeps an in-flight approval bound to the renamed room', async () => {
    const room = await loadRoom({
      approvalUntil: { ops: { payload: APPROVAL, until: 3 } },
      onResumePoll: (polls: number) => {
        if (polls === 1) {
          const all = { ...room.chat.$groupChats.get() }
          const oldRoom = all.Core || {}
          delete all.Core
          all.Renamed = { ...oldRoom }
          room.chat.$groupChats.set(all)
          room.turns.clearGroupClarify('Core')
        }
      },
      turn: () => 'approval under renamed room'
    })

    room.chat.updateGroupChat('Core', current => {
      current.roomId = 'r-core'

      return current
    })
    await room.turns.runGroupChatMemberTurn('Core', { name: 'ops', title: '' }, 'clean it', 't1', [])

    expect(Object.keys(room.chat.$groupClarify.get())).not.toContain('Core::ops')

    const remaining = Object.values(room.chat.$groupClarify.get())
    expect(remaining.filter(p => p.group === 'Renamed')).toHaveLength(0)
  })
})

describe('stranded harvest', () => {
  const member: GroupMember = { name: 'research', title: '' }

  const setup = async (room: Room, finalReply: string) => {
    room.chat.updateGroupChat('Core', current => {
      current.roomId = 'r-core'
      current.stranded = { research: { before: 1, thread: 't1' } }

      return current
    })
    await room.turns.ensureGroupChatSession('Core', member)

    const session = [...room.gateway.sessions.values()].find(s => s.profile === 'research')

    if (session) {
      session.messages.push({ content: 'hi', role: 'user' }, { content: finalReply, role: 'assistant' })
    }
  }

  it('posts a late reply into the room and clears the marker', async () => {
    const room = await loadRoom()
    await setup(room, 'This landed late')
    await room.turns.harvestStrandedGroupReply('Core', member)

    const posted = log(room, 'Core')

    expect(posted).toHaveLength(1)
    expect(posted[0].text).toBe('This landed late')
    expect(posted[0].thread).toBe('t1')
    expect(room.chat.$groupChats.get().Core?.stranded?.research).toBeUndefined()
  })

  it('prefers the substantive answer over a trailing synthetic (pass)', async () => {
    const room = await loadRoom()

    room.chat.updateGroupChat('Core', current => {
      current.roomId = 'r-core'
      current.stranded = { research: { before: 1, thread: 't1' } }

      return current
    })
    await room.turns.ensureGroupChatSession('Core', member)

    const session = [...room.gateway.sessions.values()].find(s => s.profile === 'research')

    if (session) {
      session.messages.push(
        { content: 'hi', role: 'user' },
        { content: 'Substantive analysis', role: 'assistant' },
        { content: '(pass)', role: 'assistant' }
      )
    }

    await room.turns.harvestStrandedGroupReply('Core', member)

    const posted = log(room, 'Core')

    expect(posted).toHaveLength(1)
    expect(posted[0].text).toBe('Substantive analysis')
  })

  it('consumes the marker without posting when the late reply is a pass', async () => {
    const room = await loadRoom()
    await setup(room, '(pass)')
    await room.turns.harvestStrandedGroupReply('Core', member)

    expect(log(room, 'Core')).toHaveLength(0)
    expect(room.chat.$groupChats.get().Core?.stranded?.research).toBeUndefined()
  })

  it('never re-submits into a member the harvest just confirmed is still running', async () => {
    const room = await loadRoom({
      pollsBusy: 1,
      turn: () => 'late finish'
    })

    room.chat.updateGroupChat('Core', current => {
      current.roomId = 'r-core'
      current.stranded = { research: { before: 1, thread: 't1' } }

      return current
    })
    await room.turns.ensureGroupChatSession('Core', member)
    await room.turns.harvestStrandedGroupReply('Core', member)

    // The harvest found the session still running and left the marker alone.\n    expect(room.chat.$groupChats.get().Core?.stranded?.research).toBeDefined()
  })
})

describe('room record', () => {
  it('persists the roomId alongside the room and survives another room's disband', async () => {
    const room = await loadRoom()

    room.chat.updateGroupChat('Core', current => {
      current.roomId = 'r-core'

      return current
    })
    room.chat.updateGroupChat('Other', current => {
      current.roomId = 'r-other'

      return current
    })
    await new Promise(resolve => setTimeout(resolve, 100))

    const saved = room.gateway.storage.get('group-chats') as Record<string, GroupChat>

    expect(saved.Core.roomId).toBe('r-core')
    expect(saved.Other.roomId).toBe('r-other')

    // Disband 'Other' — the durable map must still carry 'Core' intact.\n    const rooms = { ...room.chat.$groupChats.get() }
    rooms.Other = { ...rooms.Other, tombstone: true }
    room.chat.$groupChats.set(rooms)
    room.chat.updateGroupChat('Core', r => r)
    await new Promise(resolve => setTimeout(resolve, 100))

    const afterDisband = room.gateway.storage.get('group-chats') as Record<string, GroupChat>

    expect(Object.keys(afterDisband)).toEqual(['Core'])
    expect(afterDisband.Core.roomId).toBe('r-core')
  })
})
