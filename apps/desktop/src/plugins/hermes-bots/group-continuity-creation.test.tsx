import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as DataModule from './data'
import type { HostedRoomProbe } from './hosted-room-runtime'
import { translateBots } from './i18n-test-helper'
import type { BotMeta, RosterRow } from './types'

const mocks = vi.hoisted(() => ({
  createAutonomousHostedGroupChat: vi.fn(),
  markHostedRoomLocallyDeleted: vi.fn(),
  notify: vi.fn(),
  probeHostedRoomMembers: vi.fn(),
  saveBotMeta: vi.fn(async (_owner: unknown, _patch: unknown) => ({
    serverOutcome: 'persisted',
    serverPersisted: true
  }))
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  return {
    ...original,
    host: {
      ...original.host,
      notify: mocks.notify
    },
    usePluginI18n: () => translateBots
  }
})

vi.mock('./data', async importOriginal => {
  const original = await importOriginal<typeof DataModule>()

  return {
    ...original,
    saveBotMeta: mocks.saveBotMeta
  }
})

vi.mock('./hosted-room-runtime', () => ({
  createAutonomousHostedGroupChat: mocks.createAutonomousHostedGroupChat,
  describeHostedRoomCreationError: () => null,
  markHostedRoomLocallyDeleted: mocks.markHostedRoomLocallyDeleted,
  probeHostedRoomMembers: mocks.probeHostedRoomMembers
}))

const roster: RosterRow[] = [
  {
    connectionId: 'host-a',
    connectionLabel: 'Studio',
    name: 'research',
    remoteSource: true,
    sourceScoped: true,
    targetProfile: 'research'
  },
  {
    connectionId: 'host-a',
    connectionLabel: 'Studio',
    name: 'builder',
    remoteSource: true,
    sourceScoped: true,
    targetProfile: 'builder'
  }
]

const eligibleProbe: HostedRoomProbe = {
  attachmentParity: true,
  attachmentUnavailableMembers: [],
  eligible: true,
  capability: {
    authorityId: 'install:studio',
    connectionId: 'host-a',
    exactPeerGrantRevoke: false,
    kind: 'driver-capable',
    limits: {
      attachments: true,
      automaticFailover: false,
      crossGatewayMembers: true
    },
    persistentProcess: true,
    routeGrantFingerprint: false,
    reason: null,
    roomLink: null
  },
  capabilities: {},
  route: {
    connectionId: 'host-a',
    homeConnectionId: 'host-a',
    kind: 'single-gateway',
    limits: {
      attachments: true,
      automaticFailover: false,
      crossGatewayMembers: true
    },
    memberConnectionIds: ['host-a', 'host-a'],
    reason: null,
    remoteConnectionIds: []
  },
  routes: {
    'host-a': {
      connectionId: 'host-a',
      mode: 'remote',
      profile: 'default',
      targetProfile: 'default'
    }
  }
}

beforeAll(() => {
  Element.prototype.scrollIntoView = () => undefined
  Element.prototype.hasPointerCapture = () => false
  Element.prototype.releasePointerCapture = () => undefined
  Element.prototype.setPointerCapture = () => undefined
})

beforeEach(async () => {
  vi.clearAllMocks()
  mocks.probeHostedRoomMembers.mockResolvedValue(eligibleProbe)
  mocks.createAutonomousHostedGroupChat.mockResolvedValue({
    authorityId: 'install:studio',
    authorityEpoch: 1,
    connectionId: 'host-a',
    continuityMode: 'gateway'
  })

  const { $botMeta } = await import('./data')
  const { $groupChats } = await import('./group-chat')

  $botMeta.set({})
  $groupChats.set({})
})

afterEach(() => {
  cleanup()
})

async function renderSelectedGroup(rows: RosterRow[] = roster, onCreated?: (group: string) => void) {
  const { CreateGroupChatDialog } = await import('./create-dialog')

  const Harness = () => {
    const [open, setOpen] = useState(true)

    return <CreateGroupChatDialog onClose={() => setOpen(false)} onCreated={onCreated} open={open} roster={rows} />
  }

  render(<Harness />)

  const checkboxes = screen.getAllByRole('checkbox')

  fireEvent.click(checkboxes[0])
  fireEvent.click(checkboxes[1])

  await waitFor(() => expect(mocks.probeHostedRoomMembers).toHaveBeenCalledTimes(1))

  return screen.getByRole('button', {
    name: 'Create Group (2)'
  }) as HTMLButtonElement
}

describe('automatic Group Chat continuity', () => {
  it('shows the required empty copy when no bots exist', async () => {
    const { CreateGroupChatDialog } = await import('./create-dialog')

    await act(async () => {
      render(<CreateGroupChatDialog onClose={() => undefined} open roster={[]} />)
    })

    expect(screen.getByText('No bots yet. Create a bot first.')).toBeTruthy()
  })

  it('has no creation switch and selects hosted continuity automatically when eligible', async () => {
    const create = await renderSelectedGroup()

    expect(screen.getByText('New group chat')).toBeTruthy()
    expect(screen.getByText('Choose 2–6 Bots.')).toBeTruthy()
    expect(screen.queryByRole('switch')).toBeNull()
    await waitFor(() => expect(create.disabled).toBe(false))
    await act(async () => {
      fireEvent.click(create)
    })

    await waitFor(() => expect(mocks.createAutonomousHostedGroupChat).toHaveBeenCalledTimes(1))

    const { $groupChats } = await import('./group-chat')
    const created = Object.values($groupChats.get())[0]

    expect(created).toMatchObject({
      continuityMode: 'gateway',
      hosted: 'install:studio',
      hostedConnectionId: 'host-a',
      hostedEpoch: 1
    })
    expect(mocks.notify).not.toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining('pause when Desktop closes')
      })
    )
  })

  it('keeps Desktop continuity when hosted gateways cannot preserve attachments', async () => {
    mocks.probeHostedRoomMembers.mockResolvedValue({
      ...eligibleProbe,
      attachmentParity: false
    })
    const create = await renderSelectedGroup()

    await waitFor(() => expect(create.disabled).toBe(false))
    await act(async () => {
      fireEvent.click(create)
    })

    const { $groupChats } = await import('./group-chat')
    const created = Object.values($groupChats.get())[0]

    expect(created.continuityMode).toBe('desktop')
    expect(created.hosted ?? null).toBeNull()
    expect(mocks.createAutonomousHostedGroupChat).not.toHaveBeenCalled()
  })

  it('keeps each selected Bot on its captured gateway when the roster refreshes during creation', async () => {
    const rows: RosterRow[] = [
      {
        connectionId: 'host-a',
        connectionLabel: 'Studio',
        name: 'research',
        remoteSource: true,
        sourceScoped: true,
        targetProfile: 'research'
      },
      {
        connectionId: 'host-b',
        connectionLabel: 'VPS',
        name: 'builder',
        remoteSource: true,
        sourceScoped: true,
        targetProfile: 'builder'
      }
    ]

    mocks.probeHostedRoomMembers.mockResolvedValue({
      ...eligibleProbe,
      capabilities: {
        'host-a': eligibleProbe.capability,
        'host-b': {
          ...eligibleProbe.capability,
          authorityId: 'install:vps',
          connectionId: 'host-b'
        }
      },
      route: {
        ...eligibleProbe.route,
        kind: 'multi-gateway',
        memberConnectionIds: ['host-a', 'host-b'],
        remoteConnectionIds: ['host-b']
      },
      routes: {
        ...eligibleProbe.routes,
        'host-b': {
          connectionId: 'host-b',
          mode: 'remote',
          profile: 'default',
          targetProfile: 'default'
        }
      }
    })
    mocks.createAutonomousHostedGroupChat.mockImplementation(async () => {
      rows[1].connectionId = 'host-a'
      rows[1].connectionLabel = 'Studio'

      return {
        authorityId: 'install:studio',
        authorityEpoch: 1,
        connectionId: 'host-a',
        continuityMode: 'distributed'
      }
    })

    const create = await renderSelectedGroup(rows)
    await waitFor(() => expect(create.disabled).toBe(false))
    await act(async () => {
      fireEvent.click(create)
    })

    const { $groupChats } = await import('./group-chat')
    const created = Object.values($groupChats.get())[0]

    expect(created.continuityMode).toBe('distributed')
    expect(created.members).toEqual([
      expect.objectContaining({ connectionId: 'host-a', name: 'research' }),
      expect.objectContaining({ connectionId: 'host-b', name: 'builder' })
    ])
    expect(mocks.createAutonomousHostedGroupChat.mock.calls[0][0].members).toEqual([
      expect.objectContaining({
        member: expect.objectContaining({ connectionId: 'host-a', name: 'research' })
      }),
      expect.objectContaining({
        member: expect.objectContaining({ connectionId: 'host-b', name: 'builder' })
      })
    ])
    expect(mocks.saveBotMeta.mock.calls.map(([owner]) => (owner as { connectionId?: string }).connectionId)).toEqual([
      'host-a',
      'host-b'
    ])
  })

  it('keeps Create disabled until the probe settles', async () => {
    let settleProbe: (probe: HostedRoomProbe) => void = () => undefined

    mocks.probeHostedRoomMembers.mockImplementation(
      () =>
        new Promise(resolve => {
          settleProbe = resolve
        })
    )

    const create = await renderSelectedGroup()

    expect(create.disabled).toBe(true)
    settleProbe(eligibleProbe)
    await waitFor(() => expect(create.disabled).toBe(false))
  })

  it('creates a classic Desktop Group Chat when the probe fails', async () => {
    mocks.probeHostedRoomMembers.mockRejectedValue(new Error('offline'))
    const create = await renderSelectedGroup()

    await waitFor(() => expect(create.disabled).toBe(false))
    await act(async () => {
      fireEvent.click(create)
    })

    const { $groupChats } = await import('./group-chat')

    await waitFor(() => expect(Object.values($groupChats.get())).toHaveLength(1))
    const created = Object.values($groupChats.get())[0]

    expect(created.continuityMode).toBe('desktop')
    expect(created.hosted ?? null).toBeNull()
    expect(mocks.createAutonomousHostedGroupChat).not.toHaveBeenCalled()
  })

  it('keeps unscoped local metadata owners local in classic creation', async () => {
    mocks.probeHostedRoomMembers.mockRejectedValue(new Error('unsupported'))
    const create = await renderSelectedGroup([{ name: 'research' }, { name: 'builder' }])

    await waitFor(() => expect(create.disabled).toBe(false))
    await act(async () => {
      fireEvent.click(create)
    })

    const { $groupChats } = await import('./group-chat')
    const created = Object.values($groupChats.get())[0]

    expect(created.continuityMode).toBe('desktop')
    expect(
      mocks.saveBotMeta.mock.calls.map(([owner]) => ({
        name: (owner as RosterRow).name,
        remoteSource: Boolean((owner as RosterRow).remoteSource),
        sourceScoped: Boolean((owner as RosterRow).sourceScoped)
      }))
    ).toEqual([
      { name: 'research', remoteSource: false, sourceScoped: false },
      { name: 'builder', remoteSource: false, sourceScoped: false }
    ])
  })

  it('falls back to Desktop and shows one concise notice when hosted creation fails', async () => {
    mocks.createAutonomousHostedGroupChat.mockImplementation(async ({ name, roomId }) => {
      const { updateGroupChat } = await import('./group-chat')

      updateGroupChat(
        name,
        room => ({
          ...room,
          continuityMode: 'gateway',
          hosted: 'install:studio',
          hostedConnectionId: 'host-a',
          hostedEpoch: 1,
          roomId
        }),
        { sync: false }
      )

      throw new Error('device refused')
    })
    const create = await renderSelectedGroup()

    await waitFor(() => expect(create.disabled).toBe(false))
    await act(async () => {
      fireEvent.click(create)
    })

    const { $groupChats } = await import('./group-chat')

    await waitFor(() => expect(Object.values($groupChats.get())).toHaveLength(1))
    const created = Object.values($groupChats.get())[0]
    const attemptedRoomId = mocks.createAutonomousHostedGroupChat.mock.calls[0][0].roomId

    expect(created.continuityMode).toBe('desktop')
    expect(created.hosted ?? null).toBeNull()
    expect(created.roomId).not.toBe(attemptedRoomId)
    expect(mocks.markHostedRoomLocallyDeleted).toHaveBeenCalledWith(attemptedRoomId)

    const fallback = mocks.notify.mock.calls.filter(([payload]) =>
      String(payload?.message || '').includes('Keep Desktop open')
    )

    expect(fallback).toHaveLength(1)
    expect(fallback[0][0]).toEqual(
      expect.objectContaining({
        kind: 'info',
        message: "Studio can't keep this Group Chat running yet. Keep Desktop open."
      })
    )
  })

  it.each(['rejects', 'reports failure'] as const)(
    'finishes a created room when a later Bot details sync %s',
    async failure => {
      const onCreated = vi.fn()

      mocks.saveBotMeta.mockImplementationOnce(async (_owner, patch) => {
        const { $botMeta } = await import('./data')

        $botMeta.set({
          'host-a::research': patch as BotMeta
        })

        return {
          serverOutcome: 'persisted',
          serverPersisted: true
        }
      })

      if (failure === 'rejects') {
        mocks.saveBotMeta.mockRejectedValueOnce(new Error('VPS restarted during profiles.configure'))
      } else {
        mocks.saveBotMeta.mockResolvedValueOnce({
          serverOutcome: 'failed',
          serverPersisted: false
        })
      }

      const create = await renderSelectedGroup(roster, onCreated)

      fireEvent.change(screen.getByRole('textbox', { name: 'Group name' }), {
        target: { value: 'Workshop handoff' }
      })
      await waitFor(() => expect(create.disabled).toBe(false))
      await act(async () => {
        fireEvent.click(create)
      })

      const { $botMeta } = await import('./data')
      const { $groupChats } = await import('./group-chat')

      await waitFor(() => expect(onCreated).toHaveBeenCalledWith('Workshop handoff'))
      expect($botMeta.get()['host-a::research']?.groups).toContain('Workshop handoff')
      expect($groupChats.get()['Workshop handoff']).toMatchObject({
        continuityMode: 'gateway',
        hosted: 'install:studio',
        members: expect.arrayContaining([
          expect.objectContaining({ name: 'research' }),
          expect.objectContaining({ name: 'builder' })
        ])
      })
      expect(screen.queryByText('Could not create the Group Chat. Try again.')).toBeNull()
      expect(mocks.notify).toHaveBeenCalledWith({
        kind: 'warning',
        message: '“Workshop handoff” created with 2 bots. Some Bot details haven’t synced to your other devices.'
      })
    }
  )

  it('does not create a competing Desktop room while remote cleanup is uncertain', async () => {
    mocks.createAutonomousHostedGroupChat.mockRejectedValue(
      Object.assign(new Error('cleanup pending'), { fallbackSafe: false })
    )
    const create = await renderSelectedGroup()

    await waitFor(() => expect(create.disabled).toBe(false))
    await act(async () => {
      fireEvent.click(create)
    })

    const { $groupChats } = await import('./group-chat')

    await waitFor(() => expect(screen.getByText('Could not create the Group Chat. Try again.')).toBeTruthy())
    expect(Object.values($groupChats.get())).toHaveLength(0)
    expect(mocks.notify).not.toHaveBeenCalled()
  })
})
