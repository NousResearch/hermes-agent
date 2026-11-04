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

const { connection, gateway, requestProfile } = vi.hoisted(() => ({
  connection: { id: null as unknown },
  gateway: { state: null as unknown },
  requestProfile: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const state = atom('connecting')
  const connectionId = atom<null | string>('local')
  gateway.state = state
  connection.id = connectionId

  return {
    ...sdk,
    host: { ...sdk.host, requestProfile, state: { ...sdk.host.state, connectionId, gateway: state, profile: atom('default') } },
    usePluginI18n: () => translateBotsIn('en')
  }
})

import { refreshGatewayGroupRooms } from './canonical-group-registry'
import { renderGatewayRooms } from './gateway-group-rooms-test-harness'

const gatewayState = () => gateway.state as ReturnType<typeof atom<string>>
const connectionId = () => connection.id as ReturnType<typeof atom<null | string>>
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
  vi.clearAllMocks()
  gatewayState().set('connecting')
  connectionId().set('local')
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
