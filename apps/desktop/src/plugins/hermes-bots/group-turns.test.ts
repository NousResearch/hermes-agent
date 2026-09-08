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
    // "Group: <name>" sessions keep resolving after an upgrade.
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
    // the server's hidden + "Group: " title legacy fallback.
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

    expect(first.stored).not.toBe(second.stored)
    expect(room.gateway.sessions.get(String(first.stored))?.title).toBe('Group: r-one')
    expect(room.gateway.sessions.get(String(second.stored))?.title).toBe('Group: r-two')
  })

  it('fails closed on a transient resume failure instead of forking the member session', async () => {
    // Mirrors findExistingCanonicalChat's fix (87b645f52c): a resume that
    // fails for any reason OTHER than "genuinely doesn't exist" (JSON-RPC code
    // 4007) must surface, not be read as "no session, mint a new one" — that
    // would fork the member's real history and silently overwrite
    // room.sessions[key] so the old session becomes unreachable.
    const room = await loadRoom()
    const member: GroupMember = { name: 'research', title: '' }

    room.chat.updateGroupChat('Alpha', current => {
      current.roomId = 'r-one'

      return current
    })
    const first = await room.turns.ensureGroupChatSession('Alpha', member)
    const before = room.chat.$groupChats.get().Alpha.sessions

    // A real gateway error on the NEXT resume of the member's own stored
    // session — backend still warming up, a network blip, not "not found".
    const request = host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>

    host.request = async (method: string, params: Record<string, unknown>) => {
      if (method === 'session.resume' && params.session_id === first.stored) {
        throw Object.assign(new Error('gateway temporarily unavailable'), { code: 5000 })
      }

      return request(method, params)
    }

    await expect(room.turns.ensureGroupChatSession('Alpha', member)).rejects.toThrow(/Could not check .*group session/)

    host.request = request

    // Nothing was forked: the room's session pointer is untouched, and the
    // original session is still the only one on record for this member.
    expect(room.chat.$groupChats.get().Alpha.sessions).toEqual(before)
    expect(room.gateway.sessions.size).toBe(1)
  })

  it('falls through a genuine 4007 to the title lookup, then creates', async () => {
    const room = await loadRoom()

    room.chat.updateGroupChat('Alpha', current => {
      current.roomId = 'r-one'
      // A stored sid pointing at a session the gateway no longer has (e.g. an
      // out-of-band deletion) is a genuine 4007 on the first target.
      current.sessions = { research: 'sid-gone' }

      return current
    })

    const result = await room.turns.ensureGroupChatSession('Alpha', { name: 'research', title: '' })

    expect(result.stored).toBeTruthy()
    expect(result.stored).not.toBe('sid-gone')
  })
})

describe('session-gone classification', () => {
  it('treats 4001 and "not in memory" as recoverable, 4007 as not', async () => {
    const { turns } = await loadRoom()

    expect(turns.isSessionGoneError(Object.assign(new Error('x'), { code: 4001 }))).toBe(true)
    expect(turns.isSessionGoneError(new Error('session_id=rt-1 not in memory'))).toBe(true)
    expect(turns.isSessionGoneError(Object.assign(new Error('session not found'), { code: 4007 }))).toBe(false)
    expect(turns.isSessionGoneError(null)).toBe(false)
    expect(turns.isSessionGoneError(new Error('network blip'))).toBe(false)
  })

  it('recovers a 4001 on the first submit via the STORED id and delivers', async () => {
    const room = await loadRoom({
      failFirstSubmitWith: Object.assign(new Error("session-scoped RPC rejected: session_id='rt-1' not in memory"), {
        code: 4001
      }),
      turn: () => 'recovered reply'
    })

    const reply = await room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'hi there', 't1', [])

    expect(reply).toBe('recovered reply')
    // One failed submit + exactly one retry — never more.
    expect(room.gateway.rpcFor('prompt.submit')).toHaveLength(2)
    // The recovery re-resumed the durable stored id, not the dead runtime id.
    expect(room.chat.$groupChats.get().Room.sessions?.helper).toBeTruthy()
  })

  it('does not retry a persistent non-4001 submit failure', async () => {
    const room = await loadRoom({ failEverySubmitWith: new Error('backend exploded') })

    await expect(room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'hi', 't1', [])).rejects.toThrow(
      'backend exploded'
    )
    expect(room.gateway.rpcFor('prompt.submit')).toHaveLength(1)
  })
})

describe('per-turn socket lease', () => {
  it('is acquired before any session RPC and held across attach + submit', async () => {
    const room = await loadRoom({ turn: () => 'routed reply' })

    const reply = await room.turns.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'look at this', 't1', [IMG])

    expect(reply).toBe('routed reply')
    // The retain landed before the first session-scoped RPC on the route.
    expect(room.gateway.timeline[0]).toBe('retain')

    // The socket was NEVER disposed mid-turn: after every per-request lease
    // released, the turn lease still held the refcount above zero.
    for (const call of room.gateway.rpc) {
      expect(call.refcountAfter).toBeGreaterThanOrEqual(1)
    }

    // Exactly one disposal, via the turn lease's own release at the end.
    expect(room.gateway.disposals()).toBe(1)
    expect(room.gateway.timeline.at(-1)).toBe('release')
  })

  it('is released after the turn — the refcount returns to zero', async () => {
    const room = await loadRoom({ turn: () => 'done' })

    await room.turns.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'hi', 't1', [])

    expect(room.gateway.refcount()).toBe(0)
    expect(room.gateway.disposals()).toBe(1)
  })

  it('is released even when the turn fails', async () => {
    const room = await loadRoom({ failEverySubmitWith: new Error('backend exploded') })

    await expect(room.turns.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'hi', 't1', [])).rejects.toThrow()
    expect(room.gateway.refcount()).toBe(0)
  })

  it('is feature-detected: hosts without retainProfile still run the turn', async () => {
    const room = await loadRoom({ turn: () => 'legacy ok' })
    delete host.retainProfile

    expect(await room.turns.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'hi', 't1', [])).toBe('legacy ok')
  })
})

describe('push-woken poll', () => {
  it("wakes on the member session's message.complete instead of sleeping out the backstop", async () => {
    // Real timers here: the contract is about WHEN the poll re-reads.
    vi.unstubAllGlobals()
    const listeners = new Map<string, Set<(event: unknown) => void>>()
    const room = await loadRoom({ pollsBusy: 1, turn: () => 'woken reply' })

    host.onEvent = (type: string, listener: (event: unknown) => void) => {
      const set = listeners.get(type) ?? new Set()
      set.add(listener)
      listeners.set(type, set)

      return () => set.delete(listener)
    }

    const started = Date.now()
    const turn = room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'hi', 't1', [])

    // Let the submit land and the first poll wait attach its listeners, then
    // fire the terminal frame for the runtime id the submit used.
    await new Promise(resolve => setTimeout(resolve, 50))
    // The harness mints a fresh runtime id on every resume; the frame carries
    // whichever one the session currently answers to.
    const runtime = room.gateway.sessions.get(room.gateway.calls[0]?.stored)?.runtime
    expect(listeners.get('message.complete')?.size).toBe(1)

    for (const listener of listeners.get('message.complete') ?? []) {
      listener({ type: 'message.complete', session_id: runtime })
    }

    expect(await turn).toBe('woken reply')
    // Two quick re-reads (busy once, then done) — well under one 5s backstop tick.
    expect(Date.now() - started).toBeLessThan(2000)
    // Every listener was disposed once the turn finished.
    expect(listeners.get('message.complete')?.size ?? 0).toBe(0)
    expect(listeners.get('error')?.size ?? 0).toBe(0)
  })
})

// #94376: a Codex intent-ack continuation nudge can land a substantive
// answer, then get a synthetic "(pass)" reply to the nudge itself.
describe('reply selection (#94376)', () => {
  it('surfaces a substantive answer followed by a synthetic continuation (pass)', async () => {
    const answer = "Yes. I welcomed them, and I'll review their first assignments with them."

    const room = await loadRoom({
      turn: () => [
        { content: answer, role: 'assistant' },
        {
          content:
            '[System: Continue now. Execute the required tool calls and only send your final answer after completing the task.]',
          role: 'user'
        },
        { content: '(pass)', role: 'assistant' }
      ]
    })

    expect(await room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'Did you welcome them?', 't1', [])).toBe(
      answer
    )
  })

  it('still reads a genuine pass-only turn as silent', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    expect(await room.turns.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'anything new?', 't1', [])).toBe('(pass)')
  })
})

// #90694: a member can block inside its own turn on a clarify question or a
// command approval. Those live in a hidden session the room can't see, so the
// poll mirrors them into the room and holds the turn open until they resolve.
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
    let live: Awaited<ReturnType<typeof loadRoom>> | null = null
    let sawPendingAttention = false

    const room = await loadRoom({
      clarifyUntil: { research: { payload: CLARIFY, until: 3 } },
      // The mirror pass runs while the question is still blocking — this is
      // the observable proof the gate inspected pending_clarify. Asserting
      // on $groupNeedsYou/$groupClarify AFTER the turn lands proves nothing:
      // the clarify has already resolved and its mirror is gone by then.
      onResumePoll: () => {
        sawPendingAttention =
          sawPendingAttention || live!.turns.groupHasPendingClarify(live!.chat.$groupClarify.get(), 'Core')
      },
      turn: () => 'targeting staging'
    })

    live = room

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
    expect(sawPendingAttention).toBe(true)
    // Resolved and mirrored away — nothing left to badge.
    expect(room.turns.groupHasPendingClarify(room.chat.$groupClarify.get(), 'Core')).toBe(false)
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
    // Badge is derived from $groupClarify, not a copy — nothing writes
    // $groupNeedsYou here, so there is nothing to keep in sync.
    expect(turns.groupHasPendingClarify(chat.$groupClarify.get(), 'Core')).toBe(true)

    // Same request again: no new entry, identity preserved.
    turns.syncGroupClarify('Core', member, { pending_clarify: CLARIFY })

    expect(Object.values(chat.$groupClarify.get())[0]).toBe(mirrored[0])

    // Question resolved server-side: the mirror clears, and so does the
    // derived badge — no separate cleanup path required.
    expect(turns.syncGroupClarify('Core', member, {})).toBe(false)
    expect(Object.keys(chat.$groupClarify.get())).toHaveLength(0)
    expect(turns.groupHasPendingClarify(chat.$groupClarify.get(), 'Core')).toBe(false)
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

  it('clears only the disbanded room’s mirrored questions', async () => {
    const room = await loadRoom()

    room.turns.syncGroupClarify('Core', { name: 'research' }, { pending_clarify: CLARIFY })
    room.turns.syncGroupClarify('Other', { name: 'ops' }, { pending_clarify: { ...CLARIFY, request_id: 'req-2' } })
    room.turns.clearGroupClarify('Core')

    const remaining = Object.values(room.chat.$groupClarify.get())

    expect(remaining).toHaveLength(1)
    expect(remaining[0].group).toBe('Other')
    // The derived badge follows $groupClarify with no separate cleanup step.
    expect(room.turns.groupHasPendingClarify(room.chat.$groupClarify.get(), 'Core')).toBe(false)
    expect(room.turns.groupHasPendingClarify(room.chat.$groupClarify.get(), 'Other')).toBe(true)
  })

  it('keeps pending prompts independent from mention attention through their lifecycle', async () => {
    const { chat, turns } = await loadRoom()
    const member = { name: 'research', title: '' }
    turns.syncGroupClarify('Core', member, { pending_clarify: CLARIFY })
    expect(chat.$groupNeedsYou.get().Core).toBeFalsy()
    turns.syncGroupClarify('Core', { name: 'ops' }, { pending_approval: APPROVAL })
    expect(Object.values(chat.$groupClarify.get())).toHaveLength(2)
    turns.syncGroupClarify('Core', member, {})
    expect(turns.groupHasPendingClarify(chat.$groupClarify.get(), 'Core')).toBe(true)
    turns.syncGroupClarify('Core', { name: 'ops' }, {})
    expect(turns.groupHasPendingClarify(chat.$groupClarify.get(), 'Core')).toBe(false)
    chat.appendGroupChatEntry('Core', { kind: 'member', name: 'research' }, '@user please review')
    turns.syncGroupClarify('Core', member, { pending_approval: APPROVAL })
    await turns.answerGroupClarify(Object.values(chat.$groupClarify.get())[0], member, 'deny')
    expect(Object.values(chat.$groupClarify.get())).toHaveLength(0)
    expect(chat.$groupNeedsYou.get().Core).toBe(true)
  })

  it('keeps late prompt snapshots on the live room and never revives a disbanded room', async ({ onTestFinished }) => {
    for (const roomId of ['stable-room', undefined]) {
      for (const disband of [false, true]) {
        const room = await loadRoom({ turn: ({ n }) => (n === 1 ? 'Completed reply' : '(pass)') })
        const view = await import('./group-chat-view')
        const member = { name: 'research', title: '' }
        room.chat.updateGroupChat('Core', current => ({
          ...current,
          roomId,
          running: true,
          epoch: 1,
          log: [{ id: 'input', at: 1, from: { kind: 'user', name: 'You' }, text: '@research check', thread: 'thread' }]
        }))
        let entered!: () => void
        let release!: () => void

        const polled = new Promise<void>(resolve => {
          entered = resolve
        })

        const held = new Promise<void>(resolve => {
          release = resolve
        })

        const original = host.request as (method: string, params: Record<string, unknown>) => Promise<any>
        let submitted = false
        let answered = false
        let polls = 0

        host.request = async (method: string, params: Record<string, unknown>) => {
          const result = await original(method, params)

          if (method === 'prompt.submit') {
            submitted = true
          }

          if (method === 'clarify.respond') {
            answered = true
          }

          if (method === 'session.resume' && submitted && !answered) {
            if (++polls === 1) {
              entered()
              await held
            }

            return { ...result, pending_clarify: CLARIFY }
          }

          return result
        }

        const drive = room.rounds.runGroupChatRounds('Core', [member], 'thread')
        await polled

        if (disband) {
          await view.disbandGroupChat('Core', [])
          release()
          await drive
          expect(Object.values(room.chat.$groupClarify.get())).toHaveLength(0)
          expect(room.chat.$groupChats.get().Core === undefined || room.chat.$groupChats.get().Core.tombstone).toBe(
            true
          )
          expect(room.chat.$groupChats.get().Core?.log || []).toHaveLength(0)
        } else {
          await view.renameGroupChat('Core', 'Renamed', [])

          const mirrored = new Promise<void>(resolve => {
            const stop = room.chat.$groupClarify.listen(entries => {
              if (Object.keys(entries).length) {
                stop()
                resolve()
              }
            })
          })

          release()
          await mirrored
          const [prompt] = Object.values(room.chat.$groupClarify.get())
          const correctRoom = prompt.group
          await room.turns.answerGroupClarify(prompt, member, 'staging')
          await drive
          expect(correctRoom).toBe('Renamed')
          expect(Object.keys(room.chat.$groupChats.get())).toEqual(['Renamed'])
          expect(room.chat.$groupChats.get().Renamed.running).toBe(false)
          expect(
            room.chat.$groupChats
              .get()
              .Renamed.log.filter(entry => entry.from.kind === 'member')
              .map(entry => entry.text)
          ).toEqual(['Completed reply'])
          expect(Object.values(room.chat.$groupClarify.get())).toHaveLength(0)
        }
      }
    }

    // Rejected member setup/submit must not publish failure cues into a new room.
    for (const continuation of [false, true]) {
      for (const rejectedMethod of ['session.resume', 'prompt.submit']) {
        const room = await loadRoom()
        const view = await import('./group-chat-view')
        const activity = await import('./group-activity')
        const data = await import('./data')
        const members = [{ name: 'research' }, { name: 'ops' }]
        room.chat.updateGroupChat('Core', current => ({
          ...current,
          roomId: 'old-rejection-room',
          running: true,
          log: continuation
            ? [
                {
                  id: 'handoff',
                  at: 1,
                  from: { kind: 'member', name: 'research' },
                  text: '@ops check',
                  thread: 'thread'
                },
                { id: 'input', at: 2, from: { kind: 'user', name: 'You' }, text: '@research check', thread: 'thread' }
              ]
            : [{ id: 'input', at: 1, from: { kind: 'user', name: 'You' }, text: 'check', thread: 'thread' }],
          watermarks: { 'thread::research': continuation ? 2 : 0 }
        }))
        // The normal responder has no delta; the earlier unanswered @ops
        // handoff is driven by the continuation phase.
        let phaseEntered!: () => void
        let release!: () => void

        const entered = new Promise<void>(resolve => {
          phaseEntered = resolve
        })

        const held = new Promise<void>(resolve => {
          release = resolve
        })

        const original = host.request as (method: string, params: Record<string, unknown>) => Promise<any>

        host.request = async (method: string, params: Record<string, unknown>) => {
          if (method === rejectedMethod) {
            phaseEntered()
            await held
            throw new Error('401 unauthorized late rejection')
          }

          return original(method, params)
        }

        const drive = room.rounds.runGroupChatRounds('Core', members, 'thread')
        await entered
        await view.disbandGroupChat('Core', [])
        room.chat.updateGroupChat('Core', current => ({
          ...current,
          roomId: 'replacement-rejection-room',
          tombstone: false
        }))
        room.chat.appendGroupChatEntry('Core', { kind: 'member', name: 'research' }, '@user replacement needs you')

        const before = structuredClone({
          rooms: room.chat.$groupChats.get(),
          activity: activity.$groupActivity.get(),
          attention: data.$botAttention.get(),
          needsYou: room.chat.$groupNeedsYou.get()
        })

        release()
        await drive
        expect({
          rooms: room.chat.$groupChats.get(),
          activity: activity.$groupActivity.get(),
          attention: data.$botAttention.get(),
          needsYou: room.chat.$groupNeedsYou.get()
        }).toEqual(before)
        expect(room.gateway.rpcFor('prompt.submit')).toHaveLength(0)
      }
    }

    const clock = vi.spyOn(Date, 'now').mockReturnValue(0)
    onTestFinished(() => clock.mockRestore())

    for (const recreate of [false, true]) {
      clock.mockReturnValue(0)
      const room = await loadRoom()
      const view = await import('./group-chat-view')
      const member = { name: 'research', title: '' }
      room.chat.updateGroupChat('Core', current => ({ ...current, roomId: 'retired-room' }))
      let entered!: () => void
      let release!: () => void

      const polled = new Promise<void>(resolve => {
        entered = resolve
      })

      const held = new Promise<void>(resolve => {
        release = resolve
      })

      const original = host.request as (method: string, params: Record<string, unknown>) => Promise<any>
      let submitted = false

      host.request = async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.resume' && submitted) {
          entered()
          await held
          throw new Error('poll rejected after deadline')
        }

        const result = await original(method, params)

        if (method === 'prompt.submit') {
          submitted = true
        }

        return result
      }

      const turn = room.turns.runGroupChatMemberTurn('Core', member, 'check', 'thread', [])
      await polled
      await view.disbandGroupChat('Core', [])

      if (recreate) {
        room.chat.updateGroupChat('Core', current => ({ ...current, roomId: 'replacement-room' }))
        room.turns.syncGroupClarify('Core', member, { pending_clarify: CLARIFY })
      }

      const roomsBefore = structuredClone(room.chat.$groupChats.get())
      const promptsBefore = structuredClone(room.chat.$groupClarify.get())
      // Cross even the hard cap while the rejected poll is still in flight.
      clock.mockReturnValue(24 * 60 * 60 * 1000)
      release()
      expect(await turn).toBeNull()
      expect(room.chat.$groupChats.get()).toEqual(roomsBefore)
      expect(room.chat.$groupClarify.get()).toEqual(promptsBefore)
    }

    // Retirement during one background harvest must fence the next member too.
    const room = await loadRoom()
    const view = await import('./group-chat-view')
    const members = [{ name: 'research' }, { name: 'ops' }]
    room.chat.updateGroupChat('Core', current => ({
      ...current,
      roomId: 'old-harvest-room',
      running: true,
      stranded: { research: 0, ops: 0 }
    }))
    let tick!: () => void
    const previousWindow = globalThis.window
    vi.stubGlobal('window', {
      setTimeout: (callback: () => void) => {
        tick = callback

        return 0
      }
    })
    onTestFinished(() => {
      vi.stubGlobal('window', previousWindow)
    })
    let entered!: () => void
    let release!: () => void

    const polled = new Promise<void>(resolve => {
      entered = resolve
    })

    const held = new Promise<void>(resolve => {
      release = resolve
    })

    let background = false
    const backgroundProfiles: unknown[] = []
    const original = host.request as (method: string, params: Record<string, unknown>) => Promise<any>

    host.request = async (method: string, params: Record<string, unknown>) => {
      if (method !== 'session.resume') {
        return original(method, params)
      }

      if (background) {
        backgroundProfiles.push(params.profile)

        if (params.profile === 'research') {
          entered()
          await held
        }
      }

      return { running: true, pending_clarify: CLARIFY }
    }

    await room.rounds.runGroupChatRounds('Core', members, 'thread')
    background = true
    tick()
    await polled
    await view.disbandGroupChat('Core', [])
    const { setImmediate } = await import('node:timers/promises')
    await setImmediate()
    room.chat.updateGroupChat('Core', current => ({
      ...current,
      roomId: 'new-harvest-room',
      stranded: { research: 0, ops: 0 }
    }))
    const roomsBefore = structuredClone(room.chat.$groupChats.get())
    const promptsBefore = structuredClone(room.chat.$groupClarify.get())
    release()
    // Let the released RPC and its background caller finish their microtasks.
    await setImmediate()
    expect(backgroundProfiles).toEqual(['research'])
    expect(room.chat.$groupChats.get()).toEqual(roomsBefore)
    expect(room.chat.$groupClarify.get()).toEqual(promptsBefore)
  })

  it('holds the turn open on a command approval too', async () => {
    let live: Awaited<ReturnType<typeof loadRoom>> | null = null
    let sawPendingAttention = false

    const room = await loadRoom({
      approvalUntil: { research: { payload: APPROVAL, until: 3 } },
      onResumePoll: () => {
        sawPendingAttention =
          sawPendingAttention || live!.turns.groupHasPendingClarify(live!.chat.$groupClarify.get(), 'Core')
      },
      turn: () => 'build cleaned'
    })

    live = room

    const thread = room.rounds.sendToGroupChat(
      'Core',
      [{ name: 'research', title: '' }],
      '@research clean up',
      null,
      []
    )

    await drain(() => Boolean(room.chat.$groupChats.get().Core?.running))

    const replies = log(room, 'Core').filter(entry => entry.thread === thread && entry.from.kind === 'member')

    expect(replies).toHaveLength(1)
    expect(replies[0].text).toBe('build cleaned')
    expect(Object.keys(room.chat.$groupClarify.get())).toHaveLength(0)
    expect(sawPendingAttention).toBe(true)
    expect(room.turns.groupHasPendingClarify(room.chat.$groupClarify.get(), 'Core')).toBe(false)
  })

  it('mirrors an approval with its kind, command and server choices', async () => {
    const { chat, turns } = await loadRoom()

    expect(
      turns.syncGroupClarify(
        'Core',
        { name: 'research', title: '' },
        {
          pending_approval: APPROVAL,
          session_id: 'rt-research-1'
        }
      )
    ).toBe(true)

    const entry = Object.values(chat.$groupClarify.get())[0]

    expect(entry.kind).toBe('approval')
    expect(entry.command).toBe('rm -rf ./build')
    expect(entry.question).toBe('Clean the build directory')
    expect(entry.choices).toEqual(['once', 'session', 'deny'])
    expect(entry.sessionId).toBe('rt-research-1')
  })

  it('falls back to once/deny when the server sends no choice set', async () => {
    const { chat, turns } = await loadRoom()

    turns.syncGroupClarify('Core', { name: 'research' }, { pending_approval: { command: 'ls', request_id: 'req-a2' } })

    expect(Object.values(chat.$groupClarify.get())[0].choices).toEqual(['once', 'deny'])
  })

  it('routes approvals through approval.respond with the session and choice', async () => {
    const room = await loadRoom()
    const member: GroupMember = { name: 'research', title: '' }

    room.turns.syncGroupClarify('Core', member, { pending_approval: APPROVAL, session_id: 'rt-research-1' })
    await room.turns.answerGroupClarify(Object.values(room.chat.$groupClarify.get())[0], member, 'once')

    expect(room.gateway.rpcFor('approval.respond').map(call => call.params)).toEqual([
      { choice: 'once', request_id: 'req-approval-1', session_id: 'rt-research-1' }
    ])
    expect(room.gateway.rpcFor('clarify.respond')).toHaveLength(0)
    expect(Object.keys(room.chat.$groupClarify.get())).toHaveLength(0)
  })

  it('lets clarify outrank approval when a snapshot carries both', async () => {
    const { chat, turns } = await loadRoom()

    turns.syncGroupClarify('Core', { name: 'research' }, { pending_approval: APPROVAL, pending_clarify: CLARIFY })

    const entry = Object.values(chat.$groupClarify.get())[0]

    expect(entry.kind).toBe('clarify')
    expect(entry.requestId).toBe('req-clarify-1')
  })
})

// A turn that outlives its deadline leaves a "stranded" marker. The member is
// still working; the next round harvests whatever landed instead of throwing
// the finished work away.
describe('stranded harvest', () => {
  const seedSession = (room: Room, stored: string, profile: string, title: string, messages: string[][]) => {
    room.gateway.sessions.set(stored, {
      messages: messages.map(([role, content]) => ({ content, role })),
      profile,
      runtime: `rt-${profile}`,
      stored,
      title
    })
  }

  it('posts a late reply into the room and clears the marker', async () => {
    const room = await loadRoom()

    room.chat.updateGroupChat('Late', current => {
      current.sessions = { research: 'sid-research' }
      current.stranded = { research: 0 }

      return current
    })
    // The member's session finished after we stopped waiting.
    seedSession(room, 'sid-research', 'research', 'Group: Late', [
      ['user', 'the turn prompt'],
      ['assistant', 'Here is the full research result, delivered late.']
    ])

    await room.turns.harvestStrandedGroupReply('Late', { name: 'research', title: '' })

    expect(log(room, 'Late')).toHaveLength(1)
    expect(log(room, 'Late')[0].from.name).toBe('research')
    expect(log(room, 'Late')[0].text).toMatch(/delivered late/)
    expect(room.chat.$groupChats.get().Late.stranded?.research).toBeUndefined()
  })

  it('prefers the substantive answer over a trailing synthetic (pass)', async () => {
    const room = await loadRoom()

    // #94376 class bug at the second call site: the late-landing turn ends
    // with a Codex intent-ack continuation nudge that gets a synthetic
    // "(pass)" — the harvest must still surface the substantive answer.
    room.chat.updateGroupChat('Rescue', current => {
      current.sessions = { research: 'sid-research' }
      current.stranded = { research: 0 }

      return current
    })
    seedSession(room, 'sid-research', 'research', 'Group: Rescue', [
      ['user', 'the turn prompt'],
      ['assistant', 'Here is the full research result, delivered late.'],
      [
        'user',
        '[System: Continue now. Execute the required tool calls and only send your final answer after completing the task.]'
      ],
      ['assistant', '(pass)']
    ])

    await room.turns.harvestStrandedGroupReply('Rescue', { name: 'research', title: '' })

    expect(log(room, 'Rescue')).toHaveLength(1)
    expect(log(room, 'Rescue')[0].text).toMatch(/delivered late/)
    expect(room.chat.$groupChats.get().Rescue.stranded?.research).toBeUndefined()
  })

  it('consumes the marker without posting when the late reply is a pass', async () => {
    const room = await loadRoom()

    room.chat.updateGroupChat('Quiet2', current => {
      current.sessions = { builder: 'sid-builder' }
      current.stranded = { builder: 2 }

      return current
    })
    seedSession(room, 'sid-builder', 'builder', 'Group: Quiet2', [
      ['user', 'p1'],
      ['user', 'prompt'],
      ['assistant', '(pass)']
    ])

    await room.turns.harvestStrandedGroupReply('Quiet2', { name: 'builder', title: '' })

    expect(log(room, 'Quiet2')).toHaveLength(0)
    expect(room.chat.$groupChats.get().Quiet2.stranded?.builder).toBeUndefined()
  })

  it('never re-submits into a member the harvest just confirmed is still running', async () => {
    // research is confirmed busy on exactly its first two session.resume calls
    // — the number of harvest-only touches the FIXED code makes across two
    // rounds. Without the responder guard research gets re-selected and picks
    // up two EXTRA resume calls of its own before its first post-resubmit
    // poll, which then reports done — so the unguarded run still finishes fast
    // while proving the resubmission happened.
    const room = await loadRoom({
      busyResumes: { research: 2 },
      turn: ({ profile }) => (profile === 'builder' ? 'builder here, all good' : '(pass)')
    })

    // The marker's pre-thread bare-number shape is still supported: presence
    // in `stranded` is what the round-loop guard checks, not the value shape.
    room.gateway.sessions.set('sid-research', {
      messages: [],
      profile: 'research',
      runtime: 'rt-research',
      stored: 'sid-research',
      title: 'Group: Grind'
    })
    room.chat.updateGroupChat('Grind', current => {
      current.log = [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'u1', text: '@research @builder status?' }]
      current.sessions = { research: 'sid-research' }
      current.stranded = { research: 0 }
      current.watermarks = { 'legacy::builder': 0, 'legacy::research': 0 }

      return current
    })

    await room.rounds.runGroupChatRounds(
      'Grind',
      [
        { name: 'research', title: '' },
        { name: 'builder', title: '' }
      ],
      'legacy'
    )

    expect(room.gateway.calls.filter(call => call.profile === 'research')).toHaveLength(0)
    // The marker survives untouched — the harvest confirmed research is still
    // running, so there is nothing to consume yet.
    expect(room.chat.$groupChats.get().Grind.stranded?.research).toBe(0)
    expect(room.gateway.calls.filter(call => call.profile === 'builder')).toHaveLength(1)
  })
})

describe('room record', () => {
  it('persists the roomId alongside the room and survives another room’s disband', async () => {
    const room = await loadRoom()

    room.chat.updateGroupChat('Keep', current => {
      current.log = [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'k1', text: 'hi' }]
      current.roomId = 'r-keep'

      return current
    })
    room.chat.updateGroupChat('Gone', current => {
      current.roomId = 'r-gone'

      return current
    })

    const rooms = { ...room.chat.$groupChats.get() }
    delete rooms.Gone
    room.chat.$groupChats.set(rooms)
    await room.chat.persistGroupChatRooms(rooms)

    const durable = (room.gateway.storage.get('group-chats') || {}) as Record<string, GroupChat>

    expect(durable.Keep.roomId).toBe('r-keep')
    expect('Gone' in durable).toBe(false)
  })
})
