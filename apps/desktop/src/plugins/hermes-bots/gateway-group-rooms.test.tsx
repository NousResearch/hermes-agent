/**
 * The Bots roster lists gateway Group Chats by itself: no Refresh button and
 * no raw transport text. Rooms appear once the socket opens; a failed read
 * keeps the last good list; a window whose connection has no registry id (a
 * legacy remote) lists none instead of "Canonical groups require an explicit
 * connection"; a backend without gateway Group Chats lists none.
 *
 * Real SDK components and stores; only the gateway transport is mocked.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { translateBotsIn } from './i18n-test-helper'

const { connection, gateway, foreground, requestProfile } = vi.hoisted(() => ({
  connection: { id: null as unknown },
  gateway: { state: null as unknown },
  foreground: { profile: null as unknown },
  requestProfile: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const state = atom('connecting')
  const connectionId = atom<null | string>('local')
  const profile = atom('default')
  gateway.state = state
  connection.id = connectionId
  foreground.profile = profile

  return {
    ...sdk,
    host: { ...sdk.host, requestProfile, state: { ...sdk.host.state, connectionId, gateway: state, profile } },
    usePluginI18n: () => translateBotsIn('en')
  }
})

import { $canonicalGroupBindings, forgetCanonicalGroup, refreshGatewayGroupRooms } from './canonical-group-registry'
import { renderGatewayRooms } from './gateway-group-rooms-test-harness'

const gatewayState = () => gateway.state as ReturnType<typeof atom<string>>
const connectionId = () => connection.id as ReturnType<typeof atom<null | string>>
const activeProfile = () => foreground.profile as ReturnType<typeof atom<string>>
const room = { room_id: 'room-1', name: 'Launch crew', members: [], updated_at: Date.now() / 1000 }

beforeEach(() => {
  requestProfile.mockImplementation(async () => {
    if (gatewayState().get() !== 'open') {
      throw new Error('Hermes gateway is not connected')
    }

    return { rooms: [room], next_offset: null }
  })
})

afterEach(() => {
  cleanup()
  $canonicalGroupBindings.set({})
  vi.clearAllMocks()
  gatewayState().set('connecting')
  connectionId().set('local')
  activeProfile().set('default')
})

it('lists rooms once the socket opens, with no Refresh button and no error text', async () => {
  renderGatewayRooms()
  await act(async () => { await Promise.resolve() })
  expect(requestProfile).not.toHaveBeenCalled()

  await act(async () => { gatewayState().set('open') })

  expect(await screen.findByRole('button', { name: 'Launch crew' })).toBeTruthy()
  expect(screen.queryByRole('button', { name: /refresh/i })).toBeNull()
  expect(screen.queryByRole('alert')).toBeNull()
})

it('keeps the last good list when a re-read fails', async () => {
  gatewayState().set('open')
  renderGatewayRooms()
  await screen.findByRole('button', { name: 'Launch crew' })

  requestProfile.mockRejectedValue(new Error('groups.list exploded'))
  await act(async () => { refreshGatewayGroupRooms() })
  await waitFor(() => expect(requestProfile.mock.calls.length).toBeGreaterThan(1))

  expect(screen.getByRole('button', { name: 'Launch crew' })).toBeTruthy()
  expect(screen.queryByText(/exploded/)).toBeNull()
})

it('lists nothing, and says nothing, on a connection with no registry id or no gateway Group Chats', async () => {
  gatewayState().set('open')
  connectionId().set(null)
  const view = renderGatewayRooms()
  await act(async () => { await Promise.resolve() })

  expect(requestProfile).not.toHaveBeenCalled()
  expect(view.container.textContent).toBe('')

  view.unmount()
  connectionId().set('local')
  requestProfile.mockRejectedValue(Object.assign(new Error('unknown method: groups.list'), { code: -32601 }))
  const older = renderGatewayRooms()
  await waitFor(() => expect(requestProfile).toHaveBeenCalled())
  expect(older.container.textContent).toBe('')
})

it('a disbanded room never comes back: not from a slow read, a failed re-read or a bind', async () => {
  gatewayState().set('open')
  renderGatewayRooms()
  await screen.findByRole('button', { name: 'Launch crew' })
  expect($canonicalGroupBindings.get()).toEqual({})

  // The gateway still answers with the room (a read issued before the disband), then fails.
  forgetCanonicalGroup({ connectionId: 'local', profile: 'default', roomId: 'room-1', roomName: 'Launch crew' })
  requestProfile.mockResolvedValueOnce({ rooms: [room], next_offset: null }).mockRejectedValue(new Error('offline'))
  await act(async () => { refreshGatewayGroupRooms() })

  await waitFor(() => expect(screen.queryByRole('button', { name: 'Launch crew' })).toBeNull())
  expect($canonicalGroupBindings.get()).toEqual({})
})

// A room of its own: the test above leaves room-1 disbanded.
const other = { ...room, room_id: 'room-2', name: 'Field notes' }

// The redesign's Group chats section is not one agent's: it names every agent whose rooms it lists,
// so a room does not leave the list when another agent becomes the active one.
it('keeps a room listed when another of the named profiles becomes the active one', async () => {
  requestProfile.mockImplementation(async (_route, _method, params: { profile: string }) => ({
    rooms: params.profile === 'default' ? [other] : [],
    next_offset: null
  }))
  renderGatewayRooms(undefined, ['default', 'ops'])
  await act(async () => { gatewayState().set('open') })
  expect(await screen.findByRole('button', { name: 'Field notes' })).toBeTruthy()

  await act(async () => { activeProfile().set('ops') })

  expect(screen.getByRole('button', { name: 'Field notes' })).toBeTruthy()
  await waitFor(() => expect(requestProfile).toHaveBeenCalledWith(expect.anything(), 'groups.list', expect.objectContaining({ profile: 'ops' })))
  expect(screen.getByRole('button', { name: 'Field notes' })).toBeTruthy()
})

it('lists only the active profile\'s rooms when no others are named, as the Bots roster does', async () => {
  requestProfile.mockImplementation(async (_route, _method, params: { profile: string }) => ({
    rooms: params.profile === 'default' ? [other] : [],
    next_offset: null
  }))
  renderGatewayRooms()
  await act(async () => { gatewayState().set('open') })
  expect(await screen.findByRole('button', { name: 'Field notes' })).toBeTruthy()

  await act(async () => { activeProfile().set('ops') })

  await waitFor(() => expect(screen.queryByRole('button', { name: 'Field notes' })).toBeNull())
})
