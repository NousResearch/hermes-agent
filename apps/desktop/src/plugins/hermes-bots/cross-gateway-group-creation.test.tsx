import type * as HermesSdk from '@hermes/plugin-sdk'
import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $canonicalGroupBindings } from './canonical-group-registry'
import { CreateGroupChatDialog } from './create-dialog'
import { CROSS_GATEWAY_GROUP_LOCALES } from './cross-gateway-locales'
import { $botMeta } from './data'
import {
  $groupChats,
  $groupChatWorkspace,
  groupChatSyncSnapshot,
  mergeGroupChatSyncSnapshots,
  mergeRemoteGroupChatSnapshotIntoRooms
} from './group-chat'
import type * as GroupChatModule from './group-chat'
import type * as GroupChatParts from './group-chat-parts'
import { GroupChatWorkspace } from './group-chat-view'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

// Group chats whose bots live on different connected machines (#141178, #142248):
// the active gateway owns only rooms of its own bots, so a selection spanning
// connections becomes a Desktop-driven room instead of a refused groups.create.

const { request, notify, openWorkspace } = vi.hoisted(() => ({ request: vi.fn(), notify: vi.fn(), openWorkspace: vi.fn() }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')

  return {
    ...sdk,
    gatewayActivationEpoch: () => 1,
    useGatewayActivationEpoch: () => 1,
    host: {
      ...sdk.host, requestProfile: request, notify, openWorkspace,
      request: (method: string, params?: Record<string, unknown>) => request(null, method, params),
      connections: vi.fn(async () => []),
      state: {
        ...sdk.host.state,
        connectionId: sdk.atom<string | null>('local'),
        profile: sdk.atom('default'),
        gateway: sdk.atom('open')
      }
    },
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => translateBots
  }
})
vi.mock('./group-chat', async importOriginal => {
  const actual = await importOriginal<typeof GroupChatModule>()

  return { ...actual, updateGroupChat: vi.fn((group, mutate) => actual.updateGroupChat(group, mutate, { sync: false })) }
})
vi.mock('./group-chat-parts', async importOriginal => ({
  ...await importOriginal<typeof GroupChatParts>(),
  GroupImageControls: () => null
}))

const connectionId = host.state.connectionId as WritableAtom<string | null>

const alpha: RosterRow = { name: 'alpha', connectionId: 'local', connectionLabel: 'This device' }
const beta: RosterRow = { name: 'beta', connectionId: 'local', connectionLabel: 'This device' }
const remote: RosterRow = { name: 'remote', connectionId: 'other-box', connectionLabel: 'Other box', remoteSource: true }
const scout: RosterRow = { name: 'scout', connectionId: 'other-box', connectionLabel: 'Other box', remoteSource: true }
const unreachable: RosterRow = { ...remote, name: 'offline', sourceReachable: false }

const desktopDriven = CROSS_GATEWAY_GROUP_LOCALES.en.desktopDriven

beforeEach(() => {
  connectionId.set('local')
  $canonicalGroupBindings.set({})
  $groupChats.set({})
  $groupChatWorkspace.set(null)
  $botMeta.set({})
  request.mockReset().mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true, seatable_profiles: ['alpha', 'beta'] }}

    if (method === 'groups.create') {return { room: { room_id: params.room_id, name: params.name, members: params.members } }}

    if (method === 'groups.list') {return { rooms: [], next_offset: null }}

    if (method === 'profiles.configure') {return {}}

    return {}
  })
  notify.mockReset()
  openWorkspace.mockReset().mockReturnValue(() => undefined)
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(() => {
  cleanup()
  $groupChats.set({})
  localStorage.clear()
})

async function pick(roster: RosterRow[], names: string[]) {
  const onCreated = vi.fn()
  const view = render(<CreateGroupChatDialog onClose={vi.fn()} onCreated={onCreated} open roster={roster} />)
  await waitFor(() => expect(request.mock.calls.map(call => call[1])).toContain('groups.capabilities'))
  await act(async () => {})

  for (const name of names) {
    const face = view.baseElement.querySelector(`[data-bot-face="${name}"]`)
    fireEvent.click(face!.closest('label')!.querySelector('[role="checkbox"]')!)
  }

  return { onCreated, view }
}

it('offers bots from every reachable connection, each with its device badge', async () => {
  const { view } = await pick([alpha, beta, remote, unreachable], [])
  const badges = [...view.baseElement.querySelectorAll('[data-device-badge]')].map(node => node.textContent)

  expect(view.baseElement.querySelector('[data-bot-face="remote"]')).not.toBeNull()
  expect(view.baseElement.querySelector('[data-bot-face="offline"]')).toBeNull()
  expect(badges).toEqual(['This device', 'This device', 'Other box'])
})

it('a selection spanning two connections creates a Desktop-driven room, never a refused groups.create', async () => {
  const { onCreated } = await pick([alpha, beta, remote], ['alpha', 'remote'])

  expect(screen.getByRole('note').textContent).toBe(desktopDriven)
  const submit = screen.getByRole('button', { name: 'Create group (2)' }) as HTMLButtonElement
  expect(submit.disabled).toBe(false)
  await act(async () => { fireEvent.click(submit) })

  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(0)
  expect($canonicalGroupBindings.get()).toEqual({})
  const [[group, origin]] = onCreated.mock.calls
  expect(origin).toBe('name')
  const members = $groupChats.get()[group].members ?? []
  expect(members.map((member: RosterRow) => [member.name, member.connectionId])).toEqual([
    ['alpha', 'local'],
    ['remote', 'other-box']
  ])
})

it('a selection on this connection alone still creates the gateway-owned room (control)', async () => {
  const { onCreated } = await pick([alpha, beta, remote], ['alpha', 'beta'])

  expect(screen.queryByRole('note')).toBeNull()
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Create group (2)' })) })

  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(1)
  expect(onCreated.mock.calls[0][1]).toBe('key')
  expect(Object.values($canonicalGroupBindings.get())).toHaveLength(1)
})

it('a selection wholly on another connection is not offered for creation from here', async () => {
  await pick([alpha, remote, scout], ['remote', 'scout'])

  expect((screen.getByRole('button', { name: 'Create group (2)' }) as HTMLButtonElement).disabled).toBe(true)
  expect(screen.getByRole('note').textContent).toBe(CROSS_GATEWAY_GROUP_LOCALES.en.selectionElsewhere('Other box'))
})

it('a room whose members span connections opens in the Desktop-driven view on a gateway-owned install', async () => {
  await act(async () => { render(<GroupChatWorkspace group="Mixed" members={[alpha, remote]} />) })

  expect(screen.queryByRole('button', { name: 'Start gateway group' })).toBeNull()
  expect(screen.getAllByRole('textbox').length).toBeGreaterThan(0)
  expect(request.mock.calls.filter(call => call[1] === 'groups.list')).toHaveLength(0)
})

it('a room created across connections stays Desktop-driven after its remote member is removed', async () => {
  const { onCreated, view } = await pick([alpha, beta, remote], ['alpha', 'beta', 'remote'])
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Create group (3)' })) })
  view.unmount()
  const [[group]] = onCreated.mock.calls

  expect($groupChats.get()[group].engine).toBe('desktop')

  // Manage groups removed the remote bot: every remaining member is on this connection.
  await act(async () => { render(<GroupChatWorkspace group={group} members={[alpha, beta]} />) })

  expect(screen.queryByRole('button', { name: 'Start gateway group' })).toBeNull()
  expect(screen.getAllByRole('textbox').length).toBeGreaterThan(0)
  expect(request.mock.calls.filter(call => call[1] === 'groups.list')).toHaveLength(0)
})

it("a room's Desktop engine rides room sync, so another Desktop keeps driving it too", () => {
  const room = { engine: 'desktop' as const, log: [{ at: 1, from: { kind: 'user' as const, name: 'You' }, id: 'm1', text: 'hi' }], members: [alpha, beta], roomId: 'r1', watermarks: {} }
  const published = groupChatSyncSnapshot({ Mixed: room })
  const merged = mergeGroupChatSyncSnapshots({ version: 3, rooms: {} }, published)

  expect(Object.values(merged.rooms)[0].engine).toBe('desktop')
  expect(mergeRemoteGroupChatSnapshotIntoRooms(merged, {}).Mixed.engine).toBe('desktop')
})
