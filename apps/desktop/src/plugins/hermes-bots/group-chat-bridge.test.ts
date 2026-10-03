import { expect, it, vi } from 'vitest'

import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

async function setup(options: Parameters<typeof createGroupGateway>[0] = {}) {
  vi.resetModules()
  runTimersInline()
  const gateway = createGroupGateway(options)
  Object.assign(host, gateway.host)
  const chat = await import('./group-chat')
  const data = await import('./data')
  const shared = await import('./shared')
  shared.setPluginCtx(scriptedStorage(gateway.storage))
  const { createGroupChatBridge } = await import('./group-chat-bridge')
  const sdk = await import('../../sdk/group-chats')
  const bridge = createGroupChatBridge()
  const unregister = sdk.registerGroupChatsProvider(bridge)
  chat.$groupChats.set({ Team: { roomId: 'stable', log: [], watermarks: {}, members: [{ name: 'research' }] } })

  return {
    bridge,
    chat,
    data,
    gateway,
    sdk: sdk.groupChats,
    dispose: () => {
      unregister()
      bridge.dispose()
    }
  }
}

it('sends through the existing singleton before any view mounts and publishes the same history', async () => {
  const room = await setup()

  try {
    expect(room.sdk.status()).toBe('loading')
    expect(room.sdk.send({ roomId: 'stable', text: 'hello' })).toEqual({ accepted: false, error: 'loading' })
    room.bridge.setReady()
    const result = room.sdk.send({ roomId: 'stable', text: 'hello' })
    expect(result).toMatchObject({ accepted: true, threadId: expect.any(String) })
    expect(room.chat.$groupChats.get().Team.log.filter(entry => entry.from.kind === 'user')).toHaveLength(1)
    await drain(() => Boolean(room.chat.$groupChats.get().Team.running))
    expect(room.sdk.getSnapshot().rooms[0].log).toEqual(
      room.chat.$groupChats.get().Team.log.map(({ images, ...entry }) => entry)
    )
  } finally {
    room.dispose()
  }
})

it('rejects stale, ambiguous, legacy and foreign-thread targets without appending, while rename keeps identity', async () => {
  const room = await setup()
  room.bridge.setReady()

  try {
    const original = room.chat.$groupChats.get().Team
    room.chat.$groupChats.set({
      Renamed: { ...original, log: [{ at: 1, text: 'old', from: { kind: 'user', name: 'You' }, thread: 'own' }] }
    })
    expect(room.sdk.send({ roomId: 'stable', text: 'reply', threadId: 'foreign' })).toEqual({
      accepted: false,
      error: 'invalid-thread'
    })
    expect(room.sdk.send({ roomId: 'stable', text: '   ' })).toEqual({ accepted: false, error: 'invalid-input' })
    const result = room.sdk.send({ roomId: 'stable', text: 'reply', threadId: 'own' })
    expect(result).toEqual({ accepted: true, threadId: 'own' })
    await drain(() => Boolean(room.chat.$groupChats.get().Renamed.running))
    room.chat.$groupChats.set({
      Renamed: { ...original, roomId: 'replacement' },
      Legacy: { ...original, roomId: null }
    })
    expect(room.sdk.send({ roomId: 'stable', text: 'stale' })).toEqual({ accepted: false, error: 'invalid-room' })
    expect(room.sdk.send({ roomId: 'Legacy', text: 'unsafe' })).toEqual({ accepted: false, error: 'invalid-room' })
    room.chat.$groupChats.set({ A: original, B: original })
    expect(room.sdk.send({ roomId: 'stable', text: 'ambiguous' })).toEqual({ accepted: false, error: 'invalid-room' })
    room.chat.$groupChats.set({ Team: { ...original, tombstone: true } })
    expect(room.sdk.send({ roomId: 'stable', text: 'deleted' })).toEqual({ accepted: false, error: 'invalid-room' })
  } finally {
    room.dispose()
  }
})

it('deduplicates a submission key without treating acceptance as completion, and rejects key reuse for different text', async () => {
  const room = await setup()
  room.bridge.setReady()

  try {
    const input = { roomId: 'stable', text: 'hello', submissionKey: 'one-click' }
    const accepted = room.sdk.send(input)
    expect(accepted.accepted).toBe(true)
    expect(room.sdk.send(input)).toEqual(accepted)
    expect(room.chat.$groupChats.get().Team.log.filter(entry => entry.from.kind === 'user')).toHaveLength(1)
    expect(room.sdk.send({ ...input, text: 'different' })).toEqual({ accepted: false, error: 'submission-conflict' })
    await drain(() => Boolean(room.chat.$groupChats.get().Team.running))
  } finally {
    room.dispose()
  }
})

it('preserves author source identities for same-name local and remote members', async () => {
  const room = await setup()
  const { groupMemberAuthor } = await import('./group-membership')

  const members = [
    {
      name: 'planner',
      title: 'Local Planner',
      connectionId: 'local',
      connectionLabel: 'This device',
      installId: 'local-install',
      sourceScoped: true
    },
    {
      name: 'planner',
      title: 'Remote Planner',
      connectionId: 'remote-id',
      connectionLabel: 'Workshop',
      installId: 'remote-install',
      sourceScoped: true,
      remoteSource: true
    }
  ]

  try {
    // Exercise both current gateway stamps and older label-only history.
    room.chat.$groupChats.set({
      Team: {
        roomId: 'stable',
        members,
        watermarks: {},
        sessions: { private: 'secret-session' },
        log: members.flatMap(member => [
          { at: 1, from: groupMemberAuthor(member), text: member.title, thread: 't' },
          {
            at: 2,
            from: { kind: 'member' as const, name: member.name, source: member.connectionLabel },
            text: member.title,
            thread: 't'
          }
        ])
      }
    })
    room.bridge.setReady()
    const snapshot = room.sdk.getSnapshot()

    // A consumer resolves author identity only from the public snapshot.
    const projected = snapshot.rooms[0].log.map(message => {
      const candidates = snapshot.rooms[0].members.filter(
        member =>
          member.name === message.from.name &&
          (message.from.gateway
            ? member.installId === message.from.gateway
            : (member.connectionLabel || member.connectionId) === message.from.source)
      )

      return candidates.length === 1 ? candidates[0].title : undefined
    })

    expect(projected).toEqual(members.flatMap(member => [member.title, member.title]))
    expect(snapshot.rooms[0].members).toEqual(members.map(({ sourceScoped, remoteSource, ...member }) => member))
    expect(snapshot.rooms[0]).not.toHaveProperty('sessions')

    for (const member of snapshot.rooms[0].members) {
      expect(member).not.toHaveProperty('route')
    }
  } finally {
    room.dispose()
  }
})

it('sanitizes malformed persisted display fields without freezing internal objects', async () => {
  const room = await setup()
  const rawText = { private: 'internal' }
  room.chat.$groupChats.set({
    Team: {
      roomId: 'stable',
      log: [{ at: Number.NaN, text: rawText as unknown as string, from: { kind: 'user', name: 'You' } }],
      watermarks: {}
    }
  })
  room.bridge.setReady()

  try {
    const entry = room.sdk.getSnapshot().rooms[0].log[0]
    expect(typeof entry.text).toBe('string')
    expect(Number.isFinite(entry.at)).toBe(true)
    expect(Object.isFrozen(rawText)).toBe(false)
  } finally {
    room.dispose()
  }
})

it('drops a late hydration pull when its owning registration was disposed', async () => {
  const room = await setup()
  let release!: (value: unknown) => void

  const pending = new Promise(resolve => {
    release = resolve
  })

  host.request = () => pending
  let live = true
  const pull = room.chat.pullGroupChatServerState('', () => live)
  live = false
  release({
    profiles: [
      {
        name: 'default',
        ui_meta: {
          'hermes-bots-groups': {
            version: 3,
            rooms: { 'id:late': { name: 'Late', roomId: 'late', log: [], members: [] } }
          }
        }
      }
    ]
  })

  try {
    expect(await pull).toBe(false)
    expect(room.chat.$groupChats.get()).not.toHaveProperty('Late')
  } finally {
    room.dispose()
  }
})

it('seats live source-qualified members and their current mention titles rather than a painted roster', async () => {
  const room = await setup()
  room.bridge.setReady()

  try {
    const route = { connectionId: 'remote', profile: 'research', targetProfile: 'research', mode: 'remote' as const }
    room.chat.$groupChats.set({
      Team: {
        roomId: 'stable',
        log: [],
        watermarks: {},
        members: [
          { name: 'research', connectionId: 'remote', remoteSource: true, sourceScoped: true },
          { name: 'other' }
        ]
      }
    })
    room.data.$lastRoster.set([
      { name: 'research', connectionId: 'remote', remoteSource: true, sourceScoped: true, route },
      { name: 'other' }
    ])
    room.data.$botMeta.set({ 'remote::research': { title: 'Current Captain' } })
    const result = room.sdk.send({ roomId: 'stable', text: '@currentcaptain please respond' })
    expect(result.accepted).toBe(true)
    await drain(() => Boolean(room.chat.$groupChats.get().Team.running))
    expect(room.gateway.calls.map(call => call.profile)).toEqual(['research'])
    expect(room.chat.$groupChats.get().Team.members?.find(member => member.name === 'research')?.connectionId).toBe(
      'remote'
    )
  } finally {
    room.dispose()
  }
})

it('keeps local acceptance distinct from a later engine failure', async () => {
  const room = await setup({ failEverySubmitWith: new Error('offline') })
  room.bridge.setReady()

  try {
    const result = room.sdk.send({ roomId: 'stable', text: 'hello' })
    expect(result.accepted).toBe(true)
    await drain(() => Boolean(room.chat.$groupChats.get().Team.running))
    expect(room.sdk.getSnapshot().rooms[0].activity.some(event => event.kind === 'failed')).toBe(true)
    expect(result.accepted).toBe(true)
  } finally {
    room.dispose()
  }
})

it('publishes immutable allowlisted snapshots and all live updates, and detaches on disposal', async () => {
  const room = await setup()
  const changed = vi.fn()
  const unsub = room.sdk.subscribe(changed)

  try {
    const loading = room.sdk.getSnapshot()
    expect(loading.rooms).toEqual([])
    room.bridge.setReady()
    expect(changed).toHaveBeenCalled()
    const activity = await import('./group-activity')
    const record = room.chat.$groupChats.get().Team
    room.chat.$groupChats.set({
      Team: {
        ...record,
        sessions: { secret: 'session-id' },
        log: [{ at: 1, from: { kind: 'user', name: 'You' }, text: 'hello', thread: 't' }]
      }
    })
    const snapshot = room.sdk.getSnapshot()
    expect(room.sdk.getSnapshot()).toBe(snapshot)
    expect(Object.isFrozen(snapshot)).toBe(true)
    expect(Object.isFrozen(snapshot.rooms[0].log[0].from)).toBe(true)
    expect(snapshot.rooms[0]).not.toHaveProperty('sessions')
    expect(() => {
      ;(snapshot.rooms[0].log as unknown[]).push('corrupt')
    }).toThrow()
    expect(record.log).toHaveLength(0)
    activity.recordGroupActivity('Team', { kind: 'failed', reason: 'offline', thread: 't' })
    expect(room.sdk.getSnapshot().rooms[0].activity).toMatchObject([{ kind: 'failed', reason: 'offline' }])
    room.chat.$groupClarify.set({
      pending: {
        at: 2,
        choices: ['yes'],
        group: 'Team',
        kind: 'approval',
        member: 'research',
        memberKey: 'research',
        multiSelect: false,
        question: 'Allow?',
        requestId: 'r',
        sessionId: 'private',
        thread: 't'
      }
    })
    expect(room.sdk.getSnapshot().rooms[0].requests).toMatchObject([{ kind: 'approval', question: 'Allow?' }])
    expect(room.sdk.getSnapshot().rooms[0].requests[0]).not.toHaveProperty('sessionId')
    room.chat.$groupNeedsYou.set({ Team: true })
    expect(room.sdk.getSnapshot().rooms[0].needsYou).toBe(true)
    room.dispose()
    const count = changed.mock.calls.length
    room.bridge.setReady()
    room.chat.$groupChats.set({})
    expect(room.sdk.status()).toBe('unavailable')
    expect(changed).toHaveBeenCalledTimes(count)
    expect(room.bridge.send({ roomId: 'stable', text: 'after disposal' })).toEqual({
      accepted: false,
      error: 'unavailable'
    })
  } finally {
    unsub()
    room.dispose()
  }
})

it('expires accepted submission keys at the documented FIFO bound', async () => {
  const room = await setup()
  room.bridge.setReady()

  try {
    const first = { roomId: 'stable', text: 'first', submissionKey: 'first' }
    expect(room.sdk.send(first).accepted).toBe(true)
    await drain(() => Boolean(room.chat.$groupChats.get().Team.running))

    for (let index = 0; index < 256; index++) {
      expect(
        room.sdk.send({ roomId: 'stable', text: `message ${index}`, submissionKey: `key-${index}` }).accepted
      ).toBe(true)
      await drain(() => Boolean(room.chat.$groupChats.get().Team.running))
    }

    // The expired key can now identify a different deliberate submission.
    expect(room.sdk.send({ ...first, text: 'after eviction' }).accepted).toBe(true)
    await drain(() => Boolean(room.chat.$groupChats.get().Team.running))
    expect(room.sdk.send({ roomId: 'stable', text: 'conflict', submissionKey: 'key-255' })).toEqual({
      accepted: false,
      error: 'submission-conflict'
    })
  } finally {
    room.dispose()
  }
})
