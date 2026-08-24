import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupActivity from './group-activity'
import type * as groupChat from './group-chat'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type * as groupTurns from './group-turns'
import type { Attachment, GroupChat, GroupMember, GroupMessage } from './types'

// The round engine: who answers a room message, in what order, with what
// delta, and how a round ends. Every contract here is about a SERIAL loop that
// can be superseded mid-flight — a second send, a stop, or a hold all have to
// take effect at a member boundary without losing finished work.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  activity: typeof groupActivity
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

  const [activity, chat, rounds, turns, shared] = await Promise.all([
    import('./group-activity'),
    import('./group-chat'),
    import('./group-rounds'),
    import('./group-turns'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { activity, chat, gateway, rounds, turns }
}

const MEMBERS: GroupMember[] = [
  { name: 'research', title: '' },
  { name: 'builder', title: '' },
  { name: 'ops', title: 'The Ops' }
]

const IMG: Attachment = { data: 'data:image/png;base64,iVBORw0KGgo=', kind: 'image', name: 'screenshot.png' }

const log = (room: Room, group: string) => room.chat.$groupChats.get()[group]?.log || []

/** Run the room's drive to completion. */
async function settle(room: Room, group: string) {
  await drain(() => Boolean(room.chat.$groupChats.get()[group]?.running))
}

beforeEach(() => {
  runTimersInline()
})

describe('routing', () => {
  it('reads (pass), pass, pass. and empty as silence, but not real text', async () => {
    const { turns } = await loadRoom()

    expect(turns.isGroupPassText('(pass)')).toBe(true)
    expect(turns.isGroupPassText('pass')).toBe(true)
    expect(turns.isGroupPassText('Pass.')).toBe(true)
    expect(turns.isGroupPassText('  ')).toBe(true)
    expect(turns.isGroupPassText('I will pass this to ops')).toBe(false)
  })

  it('answers only @-mentioned members; @everyone or no mention means all', async () => {
    const { rounds } = await loadRoom()

    const user = (text: string): GroupMessage[] =>
      [{ at: 1, from: { kind: 'user', name: 'You' }, text }] as GroupMessage[]

    expect(rounds.resolveGroupResponders(user('@builder take this one'), MEMBERS).map(m => m.name)).toEqual(['builder'])
    expect(rounds.resolveGroupResponders(user('hello team'), MEMBERS)).toHaveLength(3)
    expect(rounds.resolveGroupResponders(user('@everyone standup'), MEMBERS)).toHaveLength(3)
  })

  it('resolves display titles to the member and never matches @user against a bot', async () => {
    const { rounds } = await loadRoom()

    const parsed = rounds.parseGroupChatMentions('@theops please check, then ping @user', MEMBERS)

    expect(parsed.mentioned.has('ops')).toBe(true)
    expect(parsed.mentioned.size).toBe(1)
  })

  it('resolves @hermes to the default member', async () => {
    const { rounds } = await loadRoom()

    const members: GroupMember[] = [
      { name: 'default', title: '' },
      { name: 'builder', title: '' }
    ]

    const parsed = rounds.parseGroupChatMentions('@hermes take a look', members)

    expect(parsed.mentioned.has('default')).toBe(true)
    expect(parsed.mentioned.size).toBe(1)
  })

  it('rotates the lead speaker each round', async () => {
    const { rounds } = await loadRoom()

    expect(rounds.rotateGroupSpeakers(MEMBERS, 1).map(m => m.name)).toEqual(['builder', 'ops', 'research'])
    expect(rounds.rotateGroupSpeakers([MEMBERS[0]], 3)).toHaveLength(1)
  })

  it('pulls a member mentioned by another bot into the NEXT round', async () => {
    const room = await loadRoom({
      turn: ({ profile, prompt }) => {
        if (profile === 'research' && !prompt.includes('(you)')) {
          return 'Interesting — @builder should own this.'
        }

        return profile === 'builder' ? 'On it. OWNER: @builder.' : '(pass)'
      }
    })

    room.rounds.sendToGroupChat(
      'Core',
      [
        { name: 'research', title: '' },
        { name: 'builder', title: '' }
      ],
      '@research thoughts?'
    )
    await settle(room, 'Core')

    const lines = log(room, 'Core').map(entry => `${entry.from.name}: ${entry.text}`)

    expect(lines.some(line => line.startsWith('research:'))).toBe(true)
    expect(lines.some(line => line.startsWith('builder: On it'))).toBe(true)
  })
})

describe('round lifecycle', () => {
  it('settles when everyone passes, logging only the user message', async () => {
    const room = await loadRoom()

    room.rounds.sendToGroupChat('Quiet', MEMBERS, 'fyi, deploy went out')
    await settle(room, 'Quiet')

    expect(log(room, 'Quiet')).toHaveLength(1)
    expect(log(room, 'Quiet')[0].from.kind).toBe('user')
    // Every member took exactly one turn (round 1), then the settle exit fired.
    expect(room.gateway.calls).toHaveLength(3)
  })

  it('stops chatty members at GROUP_CHAT_MAX_MESSAGES', async () => {
    const room = await loadRoom({ turn: ({ n }) => `message ${n} — @everyone keep going` })

    room.rounds.sendToGroupChat('Loud', MEMBERS, 'go wild')
    await settle(room, 'Loud')

    const posted = log(room, 'Loud').filter(entry => entry.from.kind === 'member')

    expect(posted.length).toBeLessThanOrEqual(room.chat.GROUP_CHAT_MAX_MESSAGES)
  })

  it('treats a failed member turn as a pass, not a room error', async () => {
    const room = await loadRoom({
      turn: ({ profile }) => {
        if (profile === 'builder') {
          throw new Error('gateway hiccup')
        }

        return '(pass)'
      }
    })

    room.rounds.sendToGroupChat('Flaky', MEMBERS, 'anyone around?')
    await settle(room, 'Flaky')

    // Just the user message; no error entries.
    expect(log(room, 'Flaky')).toHaveLength(1)
  })

  it('badges needs-you when a member addresses @user, and clears it on the next user send', async () => {
    const room = await loadRoom({
      turn: ({ profile }) => (profile === 'research' ? 'Blocked on billing access — @user which account?' : '(pass)')
    })

    const member: GroupMember[] = [{ name: 'research', title: '' }]

    room.rounds.sendToGroupChat('Escalate', member, 'sort out the invoices')
    await settle(room, 'Escalate')

    expect(room.chat.$groupNeedsYou.get().Escalate).toBe(true)

    room.rounds.sendToGroupChat('Escalate', member, 'use the ops account')

    expect(room.chat.$groupNeedsYou.get().Escalate).toBe(false)
  })

  it('converts an "(empty)" member reply like the gateway does, never appending it raw', async () => {
    const room = await loadRoom({ turn: ({ profile }) => (profile === 'research' ? '(empty)' : '(pass)') })

    room.rounds.sendToGroupChat('Sentinel', [{ name: 'research', title: '' }], '@research thoughts?')
    await settle(room, 'Sentinel')

    const replies = log(room, 'Sentinel').filter(entry => entry.from.kind === 'member')

    expect(replies).toHaveLength(1)
    expect(replies[0].text).toContain('The model returned no response after processing tool results')
    expect(replies[0].text).not.toContain('(empty)')
  })

  it('leaves normal replies untouched', async () => {
    const room = await loadRoom({ turn: ({ profile }) => (profile === 'research' ? 'I am not empty.' : '(pass)') })

    room.rounds.sendToGroupChat('Sentinel2', [{ name: 'research', title: '' }], '@research hi')
    await settle(room, 'Sentinel2')

    const replies = log(room, 'Sentinel2').filter(entry => entry.from.kind === 'member')

    expect(replies).toHaveLength(1)
    expect(replies[0].text).toBe('I am not empty.')
  })
})

describe('per-member delta', () => {
  it('feeds a second send only the NEW messages', async () => {
    const room = await loadRoom()
    const member: GroupMember[] = [{ name: 'research', title: '' }]

    room.rounds.sendToGroupChat('Delta', member, 'first message')
    await settle(room, 'Delta')
    const firstCount = room.gateway.calls.length
    room.rounds.sendToGroupChat('Delta', member, 'second message')
    await settle(room, 'Delta')

    const second = room.gateway.calls.slice(firstCount).find(call => call.prompt.includes('second message'))

    expect(second).toBeDefined()
    expect(second?.prompt).not.toContain('first message')
  })

  it('keeps concurrent rooms sharing one member isolated in sessions, deltas and context', async () => {
    const room = await loadRoom()
    const shared: GroupMember[] = [{ name: 'research', title: '' }]
    const quiet = () => !room.chat.$groupChats.get().Alpha?.running && !room.chat.$groupChats.get().Beta?.running

    // Start both rooms without waiting for either drive to finish.
    room.rounds.sendToGroupChat('Alpha', shared, 'ALPHA_ONLY_1')
    room.rounds.sendToGroupChat('Beta', shared, 'BETA_ONLY_1')
    await drain(() => !quiet())

    const alphaFirst = room.gateway.calls.find(call => call.title === 'Group: Alpha')
    const betaFirst = room.gateway.calls.find(call => call.title === 'Group: Beta')

    expect(alphaFirst && betaFirst).toBeTruthy()
    expect(alphaFirst?.stored).not.toBe(betaFirst?.stored)
    expect(alphaFirst?.runtime).not.toBe(betaFirst?.runtime)
    expect(alphaFirst?.prompt).toContain('ALPHA_ONLY_1')
    expect(alphaFirst?.prompt).not.toContain('BETA_ONLY_1')
    expect(betaFirst?.prompt).toContain('BETA_ONLY_1')
    expect(betaFirst?.prompt).not.toContain('ALPHA_ONLY_1')
    expect(room.chat.$groupChats.get().Alpha.sessions?.research).toBe(alphaFirst?.stored)
    expect(room.chat.$groupChats.get().Beta.sessions?.research).toBe(betaFirst?.stored)

    // Interleave a second pair. Each room resumes its own session and receives
    // only its unseen room delta, never the sibling room's messages.
    const firstCallCount = room.gateway.calls.length
    room.rounds.sendToGroupChat('Alpha', shared, 'ALPHA_ONLY_2')
    room.rounds.sendToGroupChat('Beta', shared, 'BETA_ONLY_2')
    await drain(() => !quiet())

    const second = room.gateway.calls.slice(firstCallCount)
    const alphaSecond = second.find(call => call.title === 'Group: Alpha')
    const betaSecond = second.find(call => call.title === 'Group: Beta')

    expect(alphaSecond?.stored).toBe(alphaFirst?.stored)
    expect(betaSecond?.stored).toBe(betaFirst?.stored)
    expect(alphaSecond?.prompt).toContain('ALPHA_ONLY_2')
    expect(alphaSecond?.prompt).not.toContain('ALPHA_ONLY_1')
    expect(alphaSecond?.prompt).not.toContain('BETA_ONLY_2')
    expect(betaSecond?.prompt).toContain('BETA_ONLY_2')
    expect(betaSecond?.prompt).not.toContain('BETA_ONLY_1')
    expect(betaSecond?.prompt).not.toContain('ALPHA_ONLY_2')

    const alphaSession = room.gateway.sessions.get(String(alphaFirst?.stored))
    const betaSession = room.gateway.sessions.get(String(betaFirst?.stored))

    expect(alphaSession?.messages.some(message => message.content.includes('BETA_ONLY'))).toBe(false)
    expect(betaSession?.messages.some(message => message.content.includes('ALPHA_ONLY'))).toBe(false)
  })
})

describe('threads', () => {
  it('mints a new thread per composer send and lands replies in it', async () => {
    const room = await loadRoom()
    const member: GroupMember[] = [{ name: 'research', title: '' }]

    const first = room.rounds.sendToGroupChat('Rooms', member, 'first topic')
    await settle(room, 'Rooms')
    const second = room.rounds.sendToGroupChat('Rooms', member, 'second topic')
    await settle(room, 'Rooms')

    expect(first).toBeTruthy()
    expect(second).toBeTruthy()
    expect(first).not.toBe(second)
    expect(log(room, 'Rooms')[0].thread).toBe(first)
    expect(log(room, 'Rooms')[1].thread).toBe(second)
  })

  it('continues an explicit thread and scopes the member delta to it', async () => {
    const room = await loadRoom({
      turn: ({ prompt }) => (prompt.includes('billing') ? 'On the billing fix.' : '(pass)')
    })

    const member: GroupMember[] = [{ name: 'research', title: '' }]

    const billing = room.rounds.sendToGroupChat('Scoped', member, 'fix the billing bug')
    await settle(room, 'Scoped')
    room.rounds.sendToGroupChat('Scoped', member, 'research pricing')
    await settle(room, 'Scoped')
    const again = room.rounds.sendToGroupChat('Scoped', member, 'billing follow-up: ship it', billing)
    await settle(room, 'Scoped')

    const followUp = room.gateway.calls.find(call => call.prompt.includes('ship it'))
    const replies = log(room, 'Scoped').filter(entry => entry.from.kind === 'member')

    expect(again).toBe(billing)
    expect(followUp).toBeDefined()
    expect(followUp?.prompt).not.toContain('research pricing')
    expect(replies.length).toBeGreaterThanOrEqual(1)
    expect(replies.every(entry => entry.thread === billing)).toBe(true)
  })
})

describe('turn prompt', () => {
  it('addresses the default profile as @hermes', async () => {
    const { rounds } = await loadRoom()

    const members: GroupMember[] = [
      { name: 'default', title: '' },
      { name: 'builder', title: '' }
    ]

    const own = rounds.buildGroupChatTurnPrompt({
      deltaLines: [],
      groupName: 'Core',
      members,
      viewer: { name: 'default', title: '' }
    })

    expect(own).toMatch(/You are @hermes,/)
    expect(own).not.toMatch(/@default\b/)

    const peer = rounds.buildGroupChatTurnPrompt({
      deltaLines: [],
      groupName: 'Core',
      members,
      viewer: { name: 'builder', title: '' }
    })

    expect(peer).toMatch(/group chat with @hermes/)
  })

  it('asks for full-quality results and short chatter, not short results', async () => {
    const { rounds } = await loadRoom()

    const prompt = rounds.buildGroupChatTurnPrompt({
      deltaLines: [],
      groupName: 'Core',
      members: [
        { name: 'research', title: '' },
        { name: 'builder', title: '' }
      ],
      viewer: { name: 'research', title: '' }
    })

    expect(prompt).toMatch(/never thin out real content/i)
    expect(prompt).toMatch(/Keep chatter short/i)
  })
})

describe('attachments', () => {
  it('stages them into EVERY responding member session before that member submits', async () => {
    const room = await loadRoom()

    const members: GroupMember[] = [
      { name: 'research', title: '' },
      { name: 'builder', title: '' }
    ]

    room.rounds.sendToGroupChat('Vision', members, 'what does this show?', null, [IMG])
    await settle(room, 'Vision')

    expect(room.gateway.attaches).toHaveLength(2)
    expect(room.gateway.attaches.map(entry => entry.profile).sort()).toEqual(['builder', 'research'])

    for (const attach of room.gateway.attaches) {
      expect(attach.filename).toBe('screenshot.png')
      expect(attach.data).toBe(IMG.data)
    }

    // Each attach landed before that member's prompt.submit: the first attach
    // happens before any submit, the second before the second submit.
    expect(room.gateway.attaches[0].order).toBe(0)
    expect(room.gateway.attaches[1].order).toBeLessThanOrEqual(1)
    expect(room.gateway.calls).toHaveLength(2)
  })

  it('scopes attachments by mention: only the mentioned member receives the image', async () => {
    const room = await loadRoom()

    room.rounds.sendToGroupChat(
      'Scoped',
      [
        { name: 'research', title: '' },
        { name: 'builder', title: '' }
      ],
      '@builder look at this',
      null,
      [IMG]
    )
    await settle(room, 'Scoped')

    expect(room.gateway.attaches.map(entry => entry.profile)).toEqual(['builder'])
  })

  it('accepts an image-only send and carries the attachment on the room entry', async () => {
    const room = await loadRoom()

    const minted = room.rounds.sendToGroupChat(
      'Silent',
      [
        { name: 'research', title: '' },
        { name: 'builder', title: '' }
      ],
      '',
      null,
      [IMG]
    )

    await settle(room, 'Silent')

    expect(minted).toBeTruthy()
    expect(log(room, 'Silent')).toHaveLength(1)
    expect(log(room, 'Silent')[0].images?.[0].name).toBe('screenshot.png')
    // Members still got prompted (with the image staged).
    expect(room.gateway.attaches).toHaveLength(2)
    expect(room.gateway.calls).toHaveLength(2)
  })

  it('never re-attaches an old entry on a later turn', async () => {
    const room = await loadRoom()

    const members: GroupMember[] = [
      { name: 'research', title: '' },
      { name: 'builder', title: '' }
    ]

    room.rounds.sendToGroupChat('Plain', members, 'with image', null, [IMG])
    await settle(room, 'Plain')

    expect(room.gateway.attaches).toHaveLength(2)

    // The image already sits behind every member's watermark.
    room.rounds.sendToGroupChat('Plain', members, 'plain follow-up')
    await settle(room, 'Plain')

    expect(room.gateway.attaches).toHaveLength(2)
  })

  it('skips an invalid attachment and still runs the member turn text-only', async () => {
    const room = await loadRoom()

    room.rounds.sendToGroupChat(
      'Degraded',
      [
        { name: 'research', title: '' },
        { name: 'builder', title: '' }
      ],
      'look',
      null,
      [{ data: null, name: 'bad' } as unknown as Attachment]
    )
    await settle(room, 'Degraded')

    expect(room.gateway.attaches).toHaveLength(0)
    expect(room.gateway.calls).toHaveLength(2)
  })

  it('routes PDFs through pdf.attach and files through file.attach, per member', async () => {
    const room = await loadRoom()
    const pdf: Attachment = { data: 'data:application/pdf;base64,JVBERi0=', kind: 'pdf', name: 'spec.pdf' }
    const doc: Attachment = { data: 'data:text/plain;base64,aGVsbG8=', kind: 'file', name: 'notes.txt' }

    room.rounds.sendToGroupChat(
      'Mixed',
      [
        { name: 'research', title: '' },
        { name: 'builder', title: '' }
      ],
      'review these',
      null,
      [IMG, pdf, doc]
    )
    await settle(room, 'Mixed')

    const byMethod: Record<string, number> = {}

    for (const attach of room.gateway.attaches) {
      byMethod[attach.method] = (byMethod[attach.method] || 0) + 1
    }

    // 3 attachments × 2 members, each via its own RPC.
    expect(room.gateway.attaches).toHaveLength(6)
    expect(byMethod).toEqual({ 'file.attach': 2, 'image.attach_bytes': 2, 'pdf.attach': 2 })

    const staged = room.gateway.attaches.find(attach => attach.method === 'pdf.attach')

    expect(staged?.filename).toBe('spec.pdf')
    expect(staged?.data).toBe(pdf.data)
  })

  it('appends the file.attach ref_text to the member turn prompt', async () => {
    const room = await loadRoom()
    const doc: Attachment = { data: 'data:text/plain;base64,aGVsbG8=', kind: 'file', name: 'notes.txt' }

    room.rounds.sendToGroupChat(
      'Refs',
      [
        { name: 'research', title: '' },
        { name: 'builder', title: '' }
      ],
      'read the notes',
      null,
      [doc]
    )
    await settle(room, 'Refs')

    expect(room.gateway.calls).toHaveLength(2)

    for (const call of room.gateway.calls) {
      expect(call.prompt).toContain('Attached files staged in your session workspace:')
      expect(call.prompt).toContain('notes.txt → @file:attachments/notes.txt')
    }
  })

  it('names attachments in the transcript line, labelling PDFs and files distinctly', async () => {
    const { rounds } = await loadRoom()
    const pdf: Attachment = { data: 'data:application/pdf;base64,JVBERi0=', kind: 'pdf', name: 'spec.pdf' }
    const doc: Attachment = { data: 'data:text/plain;base64,aGVsbG8=', kind: 'file', name: 'notes.txt' }
    const line = (entry: Partial<GroupMessage>) => rounds.formatGroupChatLine(entry as GroupMessage, 'research')

    expect(line({ from: { kind: 'user', name: 'You' }, images: [IMG], text: 'see attached' })).toBe(
      'You (user): see attached [attached image: screenshot.png]'
    )
    expect(line({ from: { kind: 'user', name: 'You' }, text: 'plain' })).toBe('You (user): plain')
    expect(
      line({
        from: { kind: 'member', name: 'builder' },
        images: [{ data: 'data:image/png;base64,x' } as Attachment],
        text: 'made this'
      })
    ).toBe('builder: made this [attached image: image]')
    expect(line({ from: { kind: 'user', name: 'You' }, images: [pdf, doc, IMG], text: 'here' })).toBe(
      'You (user): here [attached PDF: spec.pdf] [attached file: notes.txt] [attached image: screenshot.png]'
    )
  })
})

// #93129: a bot told to stop must STAY stopped.
describe('member holds (#93129)', () => {
  it('holds the mentioned member on an explicit stop', async () => {
    const { rounds } = await loadRoom()

    for (const text of ['stop @impl', '@impl stop', '@impl please halt', 'pause @impl for now']) {
      const action = rounds.classifyGroupHoldDirective(text, ['impl'], false)

      expect([...action.hold]).toEqual(['impl'])
      expect([...action.release]).toEqual([])
    }
  })

  it('holds nobody when a stop word carries no mention', async () => {
    const { rounds } = await loadRoom()

    expect([...rounds.classifyGroupHoldDirective('stop', [], false).hold]).toEqual([])
  })

  it('still holds on "don\'t stop @x" — the documented conservative trade-off', async () => {
    const { rounds } = await loadRoom()

    expect([...rounds.classifyGroupHoldDirective("don't stop @impl", ['impl'], false).hold]).toEqual(['impl'])
  })

  it('does not trigger on "stop" inside another word', async () => {
    const { rounds } = await loadRoom()

    // \b(stop|halt|pause)\b — "unstoppable" is a different token.
    const action = rounds.classifyGroupHoldDirective('@impl unstoppable work ahead', ['impl'], false)

    expect([...action.hold]).toEqual([])
    // A plain non-stop mention releases instead (direct address overrides hold).
    expect([...action.release]).toEqual(['impl'])
  })

  it('sets a hold on stop and clears it on resume for the same member', async () => {
    const { rounds } = await loadRoom()
    const stamp = { at: 1000, byMessageId: 'm1', thread: 't1' }

    const held = rounds.applyGroupHoldDirective({}, { everyone: false, mentioned: ['impl'] }, 'stop @impl', stamp)

    expect(held.impl).toBeTruthy()
    expect(held.impl.at).toBe(1000)
    expect(
      rounds.applyGroupHoldDirective(held, { everyone: false, mentioned: ['impl'] }, '@impl resume', stamp).impl
    ).toBeUndefined()
  })

  it('releases a held member on a direct non-stop mention', async () => {
    const { rounds } = await loadRoom()

    const next = rounds.applyGroupHoldDirective(
      { impl: { at: 1, byMessageId: null, thread: null } },
      { everyone: false, mentioned: ['impl'] },
      '@impl what is your status?',
      {}
    )

    expect(next.impl).toBeUndefined()
  })

  it('releases every hold on @all resume and sets every hold on @all stop', async () => {
    const { rounds } = await loadRoom()

    expect(
      rounds.applyGroupHoldDirective(
        { docs: { at: 2 }, impl: { at: 1 } },
        { everyone: true, mentioned: [] },
        '@all resume',
        {}
      )
    ).toEqual({})

    const held = rounds.applyGroupHoldDirective({}, { everyone: true, mentioned: [] }, '@all stop', { at: 5 }, [
      'impl',
      'docs'
    ])

    expect(held.impl).toBeTruthy()
    expect(held.docs).toBeTruthy()
    expect(held.impl.at).toBe(5)
  })

  it('leaves holds untouched on an unrelated room message', async () => {
    const { rounds } = await loadRoom()
    const held = { impl: { at: 1 } }

    expect(rounds.applyGroupHoldDirective(held, { everyone: false, mentioned: [] }, 'receipt round complete', {})).toBe(
      held
    )
  })

  it("does not disturb another member's hold", async () => {
    const { rounds } = await loadRoom()

    const next = rounds.applyGroupHoldDirective(
      { impl: { at: 1 } },
      { everyone: false, mentioned: ['docs'] },
      'stop @docs',
      {
        at: 2
      }
    )

    expect(next.impl).toBeTruthy()
    expect(next.docs).toBeTruthy()
  })

  it('consumes a held skip exactly once so the loop cannot spin', async () => {
    const { rounds } = await loadRoom()

    // Fresh delta → advance to log length.
    expect(rounds.heldMemberWatermarkAdvance(3, 7)).toBe(7)
    // Already consumed → no write, no spin.
    expect(rounds.heldMemberWatermarkAdvance(7, 7)).toBeNull()
    expect(rounds.heldMemberWatermarkAdvance(9, 7)).toBeNull()
    // Unset watermark treated as 0.
    expect(rounds.heldMemberWatermarkAdvance(undefined, 2)).toBe(2)
  })
})

// #91868/#94569: a REAL stop path for group-chat rounds. Before
// stopGroupThread the loop's only cancellation primitives were the epoch bump
// (checked at member boundaries only) and #93129 holds (which skip FUTURE
// turns) — the plugin issued ZERO session.interrupt RPCs, so "stop" meant
// "wait for the in-flight member to finish its whole model turn".
describe('stopGroupThread (#91868/#94569)', () => {
  const STOP_MEMBERS: GroupMember[] = [
    { name: 'alpha', title: '' },
    { name: 'beta', title: '' },
    { name: 'gamma', title: '' }
  ]

  /** Seed a room mid-round: epoch 3, running, alpha on turn with a live
   *  session id, nobody held yet. */
  function seedRoom(room: Room, turn: null | string = 'alpha') {
    room.chat.$groupChats.set({
      Room: {
        epoch: 3,
        holds: {},
        log: [],
        members: STOP_MEMBERS,
        running: true,
        sessions: { alpha: 'live-alpha-sid' },
        turn,
        watermarks: {}
      }
    } as unknown as Record<string, GroupChat>)
  }

  it('bumps the epoch, clears running/turn, and holds every member', async () => {
    const room = await loadRoom()
    seedRoom(room)

    await room.rounds.stopGroupThread('Room', 't1', STOP_MEMBERS)

    const state = room.chat.$groupChats.get().Room

    expect(state.epoch).toBe(4)
    expect(state.running).toBe(false)
    expect(state.turn).toBeNull()

    for (const member of STOP_MEMBERS) {
      expect(state.holds?.[member.name]).toBeTruthy()
      expect(state.holds?.[member.name].thread).toBe('t1')
    }
  })

  it('interrupts the member ON TURN via its live session', async () => {
    const room = await loadRoom()
    seedRoom(room)

    await room.rounds.stopGroupThread('Room', 't1', STOP_MEMBERS)

    const interrupts = room.gateway.rpcFor('session.interrupt')

    // Exactly one — the serial loop has one member in flight.
    expect(interrupts).toHaveLength(1)
    expect(interrupts[0].params.session_id).toBe('live-alpha-sid')
  })

  it('stops a room with nobody on turn without any interrupt RPC', async () => {
    const room = await loadRoom()
    seedRoom(room, null)

    await room.rounds.stopGroupThread('Room', 't1', STOP_MEMBERS)

    expect(room.gateway.rpcFor('session.interrupt')).toHaveLength(0)
    expect(room.chat.$groupChats.get().Room.running).toBe(false)
    expect(room.chat.$groupChats.get().Room.epoch).toBe(4)
  })

  it('records a stopped activity event visible in the CURRENT run', async () => {
    const room = await loadRoom()
    seedRoom(room)

    await room.rounds.stopGroupThread('Room', 't1', STOP_MEMBERS)

    const stopped = room.activity.currentGroupActivity('Room').find(event => event.kind === 'stopped')

    // Tagged with the POST-bump epoch, so it survives the epoch filter.
    expect(stopped).toBeTruthy()
    expect(stopped?.member).toBe('You')
    expect(stopped?.thread).toBe('t1')
    // The label comes from the shared activity label map and stays plain
    // English — no hardcoded localized text (the #94570 shell shipped a 停止).
    expect(room.activity.groupActivityLabel(stopped!)).toBeTruthy()
    expect(room.activity.groupActivityLabel(stopped!)).not.toMatch(/[\u4e00-\u9fff]/)
    expect(room.activity.GROUP_ACTIVITY_GLYPHS.stopped).toBeTruthy()
  })

  it('falls back to the durable room roster when called without members', async () => {
    const room = await loadRoom()
    seedRoom(room)

    await room.rounds.stopGroupThread('Room', 't1')

    expect(Object.keys(room.chat.$groupChats.get().Room.holds || {})).toHaveLength(STOP_MEMBERS.length)
    expect(room.gateway.rpcFor('session.interrupt')).toHaveLength(1)
  })

  it('abandons an in-flight turn once a stop bumps the epoch and holds the member', async () => {
    let live: Room | null = null

    const room = await loadRoom({
      onResumePoll: polls => {
        // Fire the stop from inside the poll cadence, after the second busy
        // poll — exactly the mid-turn click the Stop button produces.
        if (polls === 2) {
          void live?.rounds.stopGroupThread('Room', 't1', [{ name: 'helper', title: '' }])
        }
      },
      pollsBusy: 50,
      turn: () => 'long answer'
    })

    live = room

    const reply = await room.turns.runGroupChatMemberTurn('Room', { name: 'helper', title: '' }, 'long task', 't1', [])

    expect(reply).toBeNull()
    // The poll loop exited promptly after the stop, not at the deadline.
    expect(room.gateway.rpcFor('session.resume').length).toBeLessThanOrEqual(6)
    expect(room.chat.$groupChats.get().Room.running).toBe(false)
  })

  it('keeps polling through an ordinary newer-send epoch bump so late work still lands', async () => {
    let live: Room | null = null

    const room = await loadRoom({
      onResumePoll: polls => {
        if (polls === 1) {
          // A newer user send bumps the epoch but holds nobody. The in-flight
          // poll must keep going so the finished reply can still be delivered
          // (the #93127 commit check decides its fate at the boundary).
          live?.chat.updateGroupChat('Room', current => {
            current.epoch = (current.epoch || 0) + 1

            return current
          })
        }
      },
      pollsBusy: 3,
      turn: () => 'finished anyway'
    })

    live = room

    const reply = await room.turns.runGroupChatMemberTurn('Room', { name: 'helper', title: '' }, 'long task', 't1', [])

    expect(reply).toBe('finished anyway')
  })
})
