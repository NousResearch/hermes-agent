import { describe, expect, it } from 'vitest'

import {
  classifyHostedRoomCapability,
  createHostedRoomOutbox,
  createHostedRoomReplayState,
  deriveFriendlyHostedRoomStatus,
  isHostedRoomContinuityEligible,
  profileScopedRoomLinkEndpoint,
  reduceHostedRoomEvents,
  reduceHostedRoomOutbox,
  replayHostedRoomPages,
  resolveAutonomousRoomPlan,
  resolveSingleGatewayRoute
} from './hosted-room-client'

function event(
  seq: number,
  eventId: string,
  kind: string,
  payload: Record<string, unknown> = {},
  actor: Record<string, unknown> = {
    kind: 'gateway',
    id: 'install:home'
  }
) {
  return {
    room_id: 'room-1',
    seq,
    event_id: eventId,
    kind,
    actor,
    payload,
    created_at: seq
  }
}

describe('hosted Group Chat capability negotiation', () => {
  it('distinguishes an old gateway from a transient outage and a persistent driver', () => {
    const missing = Object.assign(new Error('JSON-RPC -32601: method not found'), {
      code: -32601
    })

    const old = classifyHostedRoomCapability(
      {
        ok: false,
        error: missing
      },
      {
        connectionId: 'gateway-a'
      }
    )

    const offline = classifyHostedRoomCapability(new Error('socket closed during reconnect'))

    const capable = classifyHostedRoomCapability(
      {
        driver: true,
        persistent_process: true,
        authority_gateway_id: 'install:home',
        features: ['peer_route_grant_fingerprint'],
        methods: ['groups.peer.revoke_exact', 'groups.attachment.put', 'groups.attachment.read'],
        max_log_limit: 250
      },
      {
        connectionId: 'gateway-a'
      }
    )

    expect(old).toMatchObject({
      kind: 'unsupported',
      reason: 'old-gateway',
      connectionId: 'gateway-a'
    })
    expect(offline.kind).toBe('transient-failure')
    expect(capable).toMatchObject({
      kind: 'driver-capable',
      authorityId: 'install:home',
      exactPeerGrantRevoke: true,
      limits: { attachments: true },
      persistentProcess: true,
      routeGrantFingerprint: true,
      maxLogLimit: 250
    })
    expect(isHostedRoomContinuityEligible(capable)).toBe(true)
    expect(
      classifyHostedRoomCapability({
        driver: true,
        persistent_process: true,
        authority_gateway_id: 'install:older'
      }).exactPeerGrantRevoke
    ).toBe(false)
    expect(
      classifyHostedRoomCapability({
        driver: true,
        persistent_process: true,
        authority_gateway_id: 'install:older'
      }).routeGrantFingerprint
    ).toBe(false)
    expect(
      isHostedRoomContinuityEligible({
        driver: true,
        persistent_process: false
      })
    ).toBe(false)
  })

  it('offers hosted continuity only when every member resolves to one gateway', () => {
    const same = resolveSingleGatewayRoute(
      [
        {
          name: 'research',
          connectionId: 'gateway-a',
          sourceScoped: true
        },
        {
          name: 'builder',
          route: {
            connectionId: 'gateway-a',
            mode: 'remote',
            profile: 'builder',
            targetProfile: 'builder'
          },
          remoteSource: true
        }
      ],
      {
        activeConnectionId: 'local'
      }
    )

    const mixed = resolveSingleGatewayRoute([
      {
        name: 'research',
        connectionId: 'gateway-a',
        sourceScoped: true
      },
      {
        name: 'builder',
        connectionId: 'gateway-b',
        sourceScoped: true
      }
    ])

    const unresolved = resolveSingleGatewayRoute(
      [
        {
          name: 'research'
        },
        {
          name: 'missing',
          sourceScoped: true
        }
      ],
      {
        activeConnectionId: 'local'
      }
    )

    expect(same).toMatchObject({
      kind: 'single-gateway',
      connectionId: 'gateway-a'
    })
    expect(mixed).toMatchObject({
      kind: 'unsupported',
      reason: 'cross-gateway'
    })
    expect(unresolved.reason).toBe('unresolved-member-route')
  })

  it('plans a direct multi-host Group Chat only from verified v2 catalogs', () => {
    const capability = (connectionId: string, installationId: string) =>
      classifyHostedRoomCapability(
        {
          authority_gateway_id: installationId,
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: {
              available: true,
              url: `https://${connectionId}.example.test:19445`
            },
            catalog: {
              attachments: false,
              catalog_digest: `digest-${connectionId}`,
              installation_id: installationId,
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [2],
              text: true
            }
          }
        },
        {
          connectionId
        }
      )

    const capabilities = {
      'host-a': capability('host-a', 'install:a'),
      'host-b': capability('host-b', 'install:b')
    }

    const plan = resolveAutonomousRoomPlan(
      [
        { connectionId: 'host-a', name: 'research', sourceScoped: true },
        { connectionId: 'host-b', name: 'builder', sourceScoped: true }
      ],
      {
        activeConnectionId: 'host-a',
        capabilities
      }
    )

    expect(plan).toMatchObject({
      connectionId: 'host-a',
      homeConnectionId: 'host-a',
      kind: 'multi-gateway',
      remoteConnectionIds: ['host-b']
    })
    expect(capabilities['host-b'].roomLink?.catalog?.attachments).toBe(false)

    const incompatible = {
      ...capabilities,
      'host-b': classifyHostedRoomCapability(
        {
          authority_gateway_id: 'install:b',
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: { available: true, url: 'https://host-b.example.test' },
            catalog: {
              catalog_digest: 'digest-b',
              installation_id: 'install:b',
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [1],
              text: true
            }
          }
        },
        { connectionId: 'host-b' }
      )
    }

    expect(
      resolveAutonomousRoomPlan(
        [
          { connectionId: 'host-a', name: 'research', sourceScoped: true },
          { connectionId: 'host-b', name: 'builder', sourceScoped: true }
        ],
        { activeConnectionId: 'host-a', capabilities: incompatible }
      )
    ).toMatchObject({
      kind: 'unsupported',
      reason: 'remote-needs-setup',
      unavailableConnectionId: 'host-b'
    })
  })

  it('scopes one advertised endpoint to the selected Bot profile', () => {
    expect(profileScopedRoomLinkEndpoint('https://peer.example.test/hermes/', 'research lead')).toBe(
      'https://peer.example.test/hermes/p/research%20lead'
    )
    expect(profileScopedRoomLinkEndpoint('https://peer.example.test/hermes/p/other', 'research lead')).toBeNull()
    expect(profileScopedRoomLinkEndpoint('https://peer.example.test/hermes/', 'default')).toBe(
      'https://peer.example.test/hermes'
    )
  })
})

describe('hosted Group Chat replay', () => {
  it('keeps only bounded opaque attachment metadata in replay', () => {
    const replayed = reduceHostedRoomEvents(createHostedRoomReplayState({ roomId: 'room-1' }), [
      event(1, 'message-1', 'message.user', {
        attachments: [
          {
            attachment_id: 'att_0123456789abcdef0123456789abcdef',
            kind: 'pdf',
            mime: 'application/pdf',
            name: 'review.pdf',
            size: 42
          },
          {
            attachment_id: '../local-path',
            kind: 'file',
            mime: 'text/plain',
            name: 'unsafe.txt',
            size: 1
          }
        ],
        text: 'Review this'
      })
    ])

    expect(replayed.messages[0].images).toEqual([
      {
        attachmentId: 'att_0123456789abcdef0123456789abcdef',
        kind: 'pdf',
        mime: 'application/pdf',
        name: 'review.pdf',
        size: 42
      }
    ])
  })

  it('normalizes gateway epoch seconds before rendering relative time', () => {
    const seconds = 1_787_968_060.355

    const replayed = reduceHostedRoomEvents(createHostedRoomReplayState({ roomId: 'room-1' }), [
      {
        ...event(1, 'message-1', 'message.user', { text: 'Hello' }),
        created_at: seconds
      }
    ])

    expect(replayed.messages[0].at).toBe(seconds * 1000)
  })

  it('orders, deduplicates, and advances across unknown event kinds without applying them', () => {
    const initial = createHostedRoomReplayState({
      roomId: 'room-1',
      name: 'Release',
      authorityId: 'install:home',
      connectionId: 'gateway-a'
    })

    const user = event(
      1,
      'message-1',
      'message.user',
      {
        text: 'Start the review',
        thread_id: 'thread-1'
      },
      {
        kind: 'user',
        id: 'desktop'
      }
    )

    const unknown = event(2, 'future-1', 'future.room.signal', {
      destructive: true
    })

    const member = event(
      3,
      'message-2',
      'message.member',
      {
        text: 'Review complete',
        thread_id: 'thread-1'
      },
      {
        kind: 'member',
        id: 'research',
        display_name: 'Research'
      }
    )

    const replayed = reduceHostedRoomEvents(initial, [member, unknown, user, user])

    expect(replayed.cursor).toBe(3)
    expect(replayed.messages.map(message => [message.seq, message.eventId, message.text])).toEqual([
      [1, 'message-1', 'Start the review'],
      [3, 'message-2', 'Review complete']
    ])
    expect(replayed.timeline.map(entry => entry.eventId)).toEqual(['message-1', 'message-2'])
    expect(reduceHostedRoomEvents(replayed, [member, user]).messages).toHaveLength(2)
  })

  it('buffers gaps and never advances past missing history', () => {
    const initial = createHostedRoomReplayState({
      roomId: 'room-1'
    })

    const later = event(2, 'message-2', 'message.member', { text: 'Done' })

    const gapped = reduceHostedRoomEvents(initial, [later])

    expect(gapped.cursor).toBe(0)
    expect(gapped.messages).toEqual([])
    expect(gapped.pendingEvents).toHaveLength(1)

    const complete = reduceHostedRoomEvents(gapped, [event(1, 'message-1', 'message.user', { text: 'Go' })])

    expect(complete.cursor).toBe(2)
    expect(complete.messages.map(message => message.text)).toEqual(['Go', 'Done'])
  })

  it('pages from the persisted cursor and stops safely on a stalled gap', async () => {
    const pages = [
      {
        events: [event(1, 'message-1', 'message.user', { text: 'Go' })],
        latest_seq: 3,
        has_more: true
      },
      {
        events: [event(3, 'message-3', 'message.member', { text: 'Done' })],
        latest_seq: 3,
        has_more: false
      }
    ]

    const replayed = await replayHostedRoomPages({
      state: createHostedRoomReplayState({
        roomId: 'room-1'
      }),
      fetchPage: async () => pages.shift()
    })

    expect(replayed).toMatchObject({
      complete: false,
      reason: 'stalled',
      pages: 2
    })
    expect(replayed.state.cursor).toBe(1)
    expect(replayed.state.pendingEvents).toHaveLength(1)
  })

  it('derives short status copy without reflecting raw provider details', () => {
    const state = reduceHostedRoomEvents(
      createHostedRoomReplayState({
        roomId: 'room-1',
        connectionId: 'gateway-a'
      }),
      [
        event(1, 'failed-1', 'turn.failed', {
          member_display_name: 'Builder',
          reason_code: 'provider_auth_or_access',
          raw_error: 'secret upstream payload'
        })
      ]
    )

    const friendly = deriveFriendlyHostedRoomStatus(state)

    expect(friendly).toMatchObject({
      kind: 'needs-attention',
      member: 'Builder',
      reasonCode: 'provider_auth_or_access'
    })
    expect(JSON.stringify(friendly)).not.toContain('secret upstream payload')
  })
})

describe('hosted Group Chat command outbox', () => {
  const command = {
    commandId: 'command-1',
    kind: 'send' as const,
    roomId: 'room-1',
    authorityId: 'install:home',
    connectionId: 'gateway-a',
    payload: {
      text: 'Hello',
      thread_id: 'thread-1'
    }
  }

  it('returns an interrupted in-flight command to pending with the same idempotency key', () => {
    const enqueued = reduceHostedRoomOutbox(createHostedRoomOutbox(), {
      type: 'enqueue',
      command
    })

    const inFlight = reduceHostedRoomOutbox(enqueued, {
      type: 'dispatch',
      commandId: command.commandId
    })

    const restored = createHostedRoomOutbox(inFlight)

    expect(restored.commands).toEqual([
      expect.objectContaining({
        commandId: command.commandId,
        status: 'pending',
        attempts: 1
      })
    ])
  })

  it('deduplicates identical commands, rejects conflicting reuse, and drops acknowledged work', () => {
    const once = reduceHostedRoomOutbox(createHostedRoomOutbox(), {
      type: 'enqueue',
      command
    })

    const twice = reduceHostedRoomOutbox(once, {
      type: 'enqueue',
      command: {
        ...command,
        payload: {
          thread_id: 'thread-1',
          text: 'Hello'
        }
      }
    })

    expect(twice.commands).toHaveLength(1)
    expect(() =>
      reduceHostedRoomOutbox(twice, {
        type: 'enqueue',
        command: {
          ...command,
          payload: {
            text: 'Different'
          }
        }
      })
    ).toThrow(/different content/)
    expect(() =>
      reduceHostedRoomOutbox(twice, {
        type: 'enqueue',
        command: {
          ...command,
          commandId: 'raw-file-command',
          payload: {
            content_base64: 'not-allowed-in-stage-1'
          }
        }
      })
    ).toThrow(/cannot carry raw attachment/)
    expect(
      reduceHostedRoomOutbox(twice, {
        type: 'acknowledge',
        commandId: command.commandId
      }).commands
    ).toEqual([])
  })

  it('fails closed before an offline device can grow the outbox without bound', () => {
    let outbox = createHostedRoomOutbox()

    for (let index = 0; index < 256; index += 1) {
      outbox = reduceHostedRoomOutbox(outbox, {
        type: 'enqueue',
        command: {
          ...command,
          commandId: `command-${index}`,
          payload: {
            text: `Message ${index}`
          }
        }
      })
    }

    expect(() =>
      reduceHostedRoomOutbox(outbox, {
        type: 'enqueue',
        command: {
          ...command,
          commandId: 'command-overflow'
        }
      })
    ).toThrow(/waiting to sync/)
  })

  it('keeps a durable failed receipt outside the active cap and retries the same command id', () => {
    const failed = reduceHostedRoomOutbox(
      reduceHostedRoomOutbox(createHostedRoomOutbox(), { type: 'enqueue', command }),
      {
        type: 'terminal-failure',
        commandId: command.commandId,
        failureCode: 'retry-exhausted'
      }
    )

    let full = failed

    for (let index = 0; index < 256; index += 1) {
      full = reduceHostedRoomOutbox(full, {
        type: 'enqueue',
        command: {
          ...command,
          commandId: `pending-${index}`,
          roomId: `room-${index + 2}`
        }
      })
    }

    expect(full.commands).toHaveLength(257)
    expect(reduceHostedRoomOutbox(failed, { type: 'retry', commandId: command.commandId }).commands[0]).toMatchObject({
      attempts: 0,
      commandId: command.commandId,
      failureCode: null,
      status: 'pending'
    })
  })
})
