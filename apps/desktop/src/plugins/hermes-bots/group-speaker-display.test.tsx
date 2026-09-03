import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as avatarModule from './avatar'
import { speakerEvent, speakerMember, speakerReplay, speakerRoom } from './group-speaker-test-fixtures'
import { translateBots } from './i18n-test-helper'
import type { GroupChat, GroupMember } from './types'
import { canonicalUser, optimisticUser, USER_TEXT, userRoom } from './user-event-test-fixtures'

const { host, sendToGroupChatDurably } = vi.hoisted(() => ({
  host: { notify: vi.fn(), requestAgent: vi.fn(), requestProfile: vi.fn() },
  sendToGroupChatDurably: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { pluginSdkMock } = await import('./group-test-utils')

  return { ...sdk, ...(await pluginSdkMock({ ...sdk.host, ...host })), usePluginI18n: () => translateBots }
})

vi.mock('./group-chat-parts', () => ({
  GroupClarifyCard: () => null,
  GroupImageControls: () => null,
  GroupMentionInput: () => null
}))

vi.mock('./group-rounds', () => ({ sendToGroupChatDurably, stopGroupThread: vi.fn() }))

vi.mock('./avatar', async importOriginal => {
  const avatar = await importOriginal<typeof avatarModule>()

  return { ...avatar, botAppearance: vi.fn(avatar.botAppearance) }
})

beforeEach(() => {
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function show(room: GroupChat, members: GroupMember[] = room.members || []) {
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')
  const { GroupRow } = await import('./bot-row')

  $groupChats.set({ Board: room })
  render(
    <>
      <GroupChatWorkspace group="Board" members={members} />
      <GroupRow active group="Board" members={members} needsYou={false} onDisband={vi.fn()} onOpen={vi.fn()} />
    </>
  )
}

describe('room speaker click and sidebar preview', () => {
  it('renders one user entry after actual room merge heals a cold optimistic/canonical cache', async () => {
    const chat = await import('./group-chat')

    const cold = JSON.parse(
      JSON.stringify(chat.durableGroupChatRooms({ Board: userRoom([optimisticUser(), canonicalUser()]) }))
    )

    const projection = chat.groupChatSyncSnapshot(cold)
    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, cold).Board

    await show(restored)

    expect(screen.getAllByText(USER_TEXT, { exact: true })).toHaveLength(1)
    expect(screen.getAllByText('You', { exact: true })).toHaveLength(1)
    expect(sendToGroupChatDurably).not.toHaveBeenCalled()
    expect(host.requestAgent).not.toHaveBeenCalled()
  })

  it('renders both genuinely sequenced user events even when their text is identical', async () => {
    const chat = await import('./group-chat')
    const canonical = userRoom([canonicalUser('older-event', 4), canonicalUser()])

    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(chat.groupChatSyncSnapshot({ Board: canonical }), {
      Board: canonical
    }).Board

    await show(restored)

    expect(screen.getAllByText(USER_TEXT, { exact: true })).toHaveLength(2)
    expect(screen.getAllByText('You', { exact: true })).toHaveLength(2)
  })

  it('reveals Product (@pm) from real replay, never the display label as a handle', async () => {
    const { $botMeta } = await import('./data')
    const { botAppearance } = await import('./avatar')

    $botMeta.set({ t2oracle: { title: 'Unrelated local bot' }, Product: { title: 'Wrong display match' } })
    await show(speakerRoom())
    fireEvent.click(screen.getByRole('button', { name: 'Product' }))

    expect(screen.getByRole('button', { name: 'Product (@pm)' })).toBeTruthy()
    expect(screen.getByText('@pm: Decision ready.')).toBeTruthy()
    expect(sendToGroupChatDurably).not.toHaveBeenCalled()
    expect(host.requestAgent).not.toHaveBeenCalled()
    expect(host.requestProfile).not.toHaveBeenCalled()
    expect(botAppearance).toHaveBeenCalledWith('t2oracle', undefined)
    expect(botAppearance).not.toHaveBeenCalledWith('Product', expect.anything())
    $botMeta.set({})
  })

  it('distinguishes two peers with the same profile and display name by member ID', async () => {
    const members = [speakerMember('ux', 'default', 'Team'), speakerMember('reviewer', 'default', 'Team')]

    await show(
      speakerRoom(
        speakerReplay([speakerEvent(1, 'ux', 'default', 'Team'), speakerEvent(2, 'reviewer', 'default', 'Team')]),
        members
      )
    )
    fireEvent.click(screen.getAllByRole('button', { name: 'Team' })[0])
    expect(screen.getByRole('button', { name: 'Team (@ux)' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Team' }))
    expect(screen.getByRole('button', { name: 'Team (@reviewer)' })).toBeTruthy()
    expect(screen.getByText('@reviewer: Decision ready.')).toBeTruthy()
  })

  it('uses the bound current member after a display rename and projection replay', async () => {
    const chat = await import('./group-chat')
    const room = speakerRoom(speakerReplay(), [speakerMember('pm', 't2oracle', 'Roadmap')])
    const projection = chat.groupChatSyncSnapshot({ Board: room })
    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, {})

    await show(restored.Board)
    fireEvent.click(screen.getByRole('button', { name: 'Roadmap' }))
    expect(screen.getByRole('button', { name: 'Roadmap (@pm)' })).toBeTruthy()
    expect(screen.getByText('@pm: Decision ready.')).toBeTruthy()
  })

  it('binds old events without actor.profile by their exact room/member ID', async () => {
    const event = speakerEvent()
    const { profile: _profile, ...actor } = event.actor

    await show(speakerRoom(speakerReplay([{ ...event, actor } as ReturnType<typeof speakerEvent>])))
    fireEvent.click(screen.getByRole('button', { name: 'Product' }))
    expect(screen.getByRole('button', { name: 'Product (@pm)' })).toBeTruthy()
    expect(screen.getByText('@pm: Decision ready.')).toBeTruthy()
  })

  it.each(['Product', 'default', 'Device pm'])('does not guess hosted legacy author %s', async name => {
    const room = speakerRoom([{ at: 1, from: { kind: 'member', name }, text: 'Old entry' }])

    await show(room)
    expect(screen.queryByTitle('Show full handle')).toBeNull()
    expect(screen.getByText(`${name}: Old entry`)).toBeTruthy()
  })

  it.each(['wrong-room', 'wrong-profile', 'duplicate-id', 'missing-handle'])(
    'degrades %s without inventing a handle',
    async scenario => {
      const member = speakerMember()

      if (scenario === 'wrong-room') {
        member.hostedIdentity!.roomId = 'other-room'
      }

      if (scenario === 'wrong-profile') {
        member.hostedIdentity!.profile = 'other-profile'
      }

      if (scenario === 'missing-handle') {
        delete member.handle
      }

      const members = scenario === 'duplicate-id' ? [member, { ...member, handle: 'other' }] : [member]

      await show(speakerRoom(speakerReplay(), members))
      expect(screen.queryByTitle('Show full handle')).toBeNull()
      expect(screen.getByText('Product: Decision ready.')).toBeTruthy()
    }
  )

  it('preserves classic default and remote custom-handle labels', async () => {
    const members: GroupMember[] = [
      { name: 'default' },
      {
        name: 'research',
        connectionId: 'remote',
        connectionLabel: 'Lab',
        remoteSource: true,
        handle: 'research-lab'
      }
    ]

    await show({
      watermarks: {},
      log: [
        { at: 1, from: { kind: 'member', name: 'default' }, text: 'Hello' },
        { at: 2, from: { kind: 'member', name: 'research', source: 'Lab' }, text: 'Classic reply' }
      ],
      members
    })
    fireEvent.click(screen.getByRole('button', { name: 'Hermes' }))
    expect(screen.getByRole('button', { name: 'Hermes (@hermes)' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Research' }))
    expect(screen.getByRole('button', { name: 'Research-Lab (@research-lab)' })).toBeTruthy()
    expect(screen.getByText('@research-lab: Classic reply')).toBeTruthy()
  })
})

describe('reconciled room speaker display', () => {
  it('renders every stable old mirror event with its own recovered speaker', async () => {
    const chat = await import('./group-chat')
    const members = [speakerMember('ux', 'default', 'Team'), speakerMember('reviewer', 'default', 'Team')]

    const log = speakerReplay([
      speakerEvent(1, 'ux', 'default', 'Team'),
      speakerEvent(2, 'reviewer', 'default', 'Team'),
      speakerEvent(3, 'reviewer', 'default', 'Team')
    ])

    const current = { Board: speakerRoom(log, members) }
    const mirror = chat.groupChatSyncSnapshot(current)

    for (const entry of mirror.rooms['id:room-1'].log) {
      delete entry.from.hostedIdentity
    }

    await show(chat.mergeRemoteGroupChatSnapshotIntoRooms(mirror, current).Board)

    expect(screen.getAllByRole('button', { name: 'Team' })).toHaveLength(3)
    fireEvent.click(screen.getAllByRole('button', { name: 'Team' })[0])
    expect(screen.getByRole('button', { name: 'Team (@ux)' })).toBeTruthy()
    expect(screen.getByText('@reviewer: Decision ready.')).toBeTruthy()
  })

  it.each([false, true])('keeps repeated persisted conflicts unbound, sequenced=%s', async sequenced => {
    const chat = await import('./group-chat')
    const members = [speakerMember('ux', 'default', 'Team'), speakerMember('reviewer', 'default', 'Team')]
    const a = speakerReplay([speakerEvent(1, 'ux', 'default', 'Team')])
    const b = speakerReplay([speakerEvent(1, 'reviewer', 'default', 'Team')])

    if (!sequenced) {
      delete a[0].seq
      delete b[0].seq
    }

    const conflicted = speakerRoom(chat.mergeGroupChatSyncEntries(a, b), members)
    const durable = JSON.parse(JSON.stringify(chat.durableGroupChatRooms({ Board: conflicted })))
    const cold = chat.mergeRemoteGroupChatSnapshotIntoRooms(chat.groupChatSyncSnapshot(durable), {}).Board
    cold.log = chat.mergeGroupChatSyncEntries(cold.log, a, b)
    await show(cold)

    expect(screen.queryByTitle('Show full handle')).toBeNull()
    expect(screen.getByText('Team: Decision ready.')).toBeTruthy()
    expect(sendToGroupChatDurably).not.toHaveBeenCalled()
    expect(host.requestProfile).not.toHaveBeenCalled()
  })

  it('never puts Product handle on a mismatched body after room hydration', async () => {
    const chat = await import('./group-chat')
    const room = speakerRoom()
    const mirror = chat.groupChatSyncSnapshot({ Board: room })
    const unrelated = mirror.rooms['id:room-1'].log[0]
    unrelated.text = 'Unrelated body'
    unrelated.at = room.log[0].at + 1000
    unrelated.from = { kind: 'member', name: 'Other author' }
    const hydrated = chat.mergeRemoteGroupChatSnapshotIntoRooms(mirror, { Board: room }).Board
    await show(hydrated)

    const message = screen.getByText('Unrelated body').closest('.group') as HTMLElement
    expect(within(message).queryByTitle('Show full handle')).toBeNull()
    expect(screen.getByText('Other author: Unrelated body')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Product' }))
    expect(screen.getByRole('button', { name: 'Product (@pm)' })).toBeTruthy()
    expect(sendToGroupChatDurably).not.toHaveBeenCalled()
    expect(host.requestAgent).not.toHaveBeenCalled()
  })
})
