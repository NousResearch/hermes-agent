/**
 * #143571: a gateway Group Chat row in the Bots roster offers Delete from its
 * right-click menu. Delete confirms, disbands the room on its own gateway
 * (`groups.disband`), drops it from the list, forgets its bindings and closes
 * a pane still showing it. A refused disband keeps the row and says why.
 *
 * Real SDK components and stores; only the gateway transport is mocked.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { translateBotsIn } from './i18n-test-helper'

const { notify, requestProfile } = vi.hoisted(() => ({ notify: vi.fn(), requestProfile: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: {
      ...sdk.host,
      notify,
      requestProfile,
      state: { ...sdk.host.state, connectionId: atom<null | string>('local'), profile: atom('default') }
    },
    usePluginI18n: () => translateBotsIn('en')
  }
})

import { $canonicalGroupBindings, CanonicalGroupList, publishCanonicalGroup } from './canonical-group-registry'
import { $groupChatWorkspace } from './group-chat'
import { dropGroupMainTab, recordGroupMainTab } from './group-panes'

const room = { room_id: 'room-1', name: 'Launch crew', members: [{ member_id: 'm1', profile: 'work', handle: 'work' }] }
let rooms: (typeof room)[] = []
let refuse = ''

beforeEach(() => {
  rooms = [room]
  refuse = ''
  requestProfile.mockImplementation(async (_route: unknown, method: string, params: { room_id?: string }) => {
    if (method === 'groups.capabilities') {
      return { driver: true }
    }

    if (method === 'groups.list') {
      return { rooms, next_offset: null }
    }

    if (method === 'groups.disband') {
      if (refuse) {
        throw new Error(refuse)
      }

      rooms = rooms.filter(r => r.room_id !== params.room_id)

      return { tombstone: { room_id: params.room_id } }
    }

    return {}
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $canonicalGroupBindings.set({})
  $groupChatWorkspace.set(null)
  dropGroupMainTab('Launch crew')
  dropGroupMainTab('Other crew')
})

async function openDeleteConfirm() {
  fireEvent.contextMenu(await screen.findByRole('button', { name: 'Launch crew' }))
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }))

  return within(await screen.findByRole('dialog'))
}

it('deletes a gateway group from its row menu: confirm, disband on its gateway, gone from the list', async () => {
  const onOpen = vi.fn()
  render(<CanonicalGroupList onOpen={onOpen} />)
  const dialog = await openDeleteConfirm()
  const key = Object.keys($canonicalGroupBindings.get()).find(k => k.startsWith('canonical:'))!
  $groupChatWorkspace.set(key)

  expect(dialog.getByText('Launch crew')).toBeTruthy()
  fireEvent.click(dialog.getByRole('button', { name: 'Disband' }))

  await waitFor(() => expect(screen.queryByRole('button', { name: 'Launch crew' })).toBeNull())
  const disband = requestProfile.mock.calls.filter(([, method]) => method === 'groups.disband')
  expect(disband).toHaveLength(1)
  expect(disband[0][0]).toMatchObject({ connectionId: 'local', profile: 'default' })
  expect(disband[0][2]).toMatchObject({ room_id: 'room-1' })
  expect($canonicalGroupBindings.get()[key]).toBeUndefined()
  expect($groupChatWorkspace.get()).toBeNull()
  expect(notify).toHaveBeenCalledWith({ kind: 'success', message: 'Disbanded “Launch crew”' })
  expect(onOpen).not.toHaveBeenCalled()
})

it('keeps the row and shows the gateway refusal when the disband fails', async () => {
  refuse = 'room is busy'
  render(<CanonicalGroupList onOpen={vi.fn()} />)
  const dialog = await openDeleteConfirm()
  fireEvent.click(dialog.getByRole('button', { name: 'Disband' }))

  expect(await dialog.findByText(/room is busy/)).toBeTruthy()
  expect(screen.getByRole('button', { hidden: true, name: 'Launch crew' })).toBeTruthy()
  expect(notify).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }))
})

it('closes the tab and selection opened under the room name, and leaves another room open', async () => {
  const other = { room_id: 'room-2', name: 'Other crew', members: [] }
  rooms = [room, other]
  const route = { connectionId: 'local', profile: 'default' }
  publishCanonicalGroup(route, room.name, { authority: route, room }, 'name')
  publishCanonicalGroup(route, other.name, { authority: route, room: other }, 'name')
  const closeAlias = vi.fn()
  const closeOther = vi.fn()
  recordGroupMainTab(room.name, closeAlias)
  recordGroupMainTab(other.name, closeOther)
  $groupChatWorkspace.set(room.name)

  render(<CanonicalGroupList onOpen={vi.fn()} />)
  fireEvent.click((await openDeleteConfirm()).getByRole('button', { name: 'Disband' }))

  await waitFor(() => expect(screen.queryByRole('button', { name: 'Launch crew' })).toBeNull())
  expect(closeAlias).toHaveBeenCalledTimes(1)
  expect($groupChatWorkspace.get()).toBeNull()
  expect(closeOther).not.toHaveBeenCalled()
  expect(Object.values($canonicalGroupBindings.get()).map(b => b.roomId)).toEqual(expect.arrayContaining(['room-2']))
  expect(Object.values($canonicalGroupBindings.get()).some(b => b.roomId === 'room-1')).toBe(false)
})
