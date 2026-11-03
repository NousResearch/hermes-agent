/**
 * An app relaunch restores the Bots pane before the gateway socket opens. The
 * gateway Group Chat list must not pin that refusal ("Hermes gateway is not
 * connected") as an alert; it lists the rooms once the socket opens.
 *
 * Real SDK components and stores; only the gateway transport is mocked.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { translateBotsIn } from './i18n-test-helper'

const { gateway, requestProfile } = vi.hoisted(() => ({ gateway: { state: null as unknown }, requestProfile: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const state = atom('connecting')
  gateway.state = state

  return {
    ...sdk,
    host: {
      ...sdk.host,
      requestProfile,
      state: { ...sdk.host.state, connectionId: atom<null | string>('local'), gateway: state, profile: atom('default') }
    },
    usePluginI18n: () => translateBotsIn('en')
  }
})

import { CanonicalGroupList } from './canonical-group-registry'

const gatewayState = () => gateway.state as ReturnType<typeof atom<string>>

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  gatewayState().set('connecting')
})

it('lists the rooms once the socket opens, without pinning the pre-open refusal', async () => {
  requestProfile.mockImplementation(async (_route: unknown, method: string) => {
    if (gatewayState().get() !== 'open') {
      throw new Error('Hermes gateway is not connected')
    }

    return method === 'groups.list'
      ? { rooms: [{ room_id: 'room-1', name: 'Launch crew', members: [] }], next_offset: null }
      : { driver: true }
  })

  render(<CanonicalGroupList onOpen={() => undefined} />)
  await act(async () => {
    await Promise.resolve()
  })
  expect(screen.queryByRole('alert')).toBeNull()

  await act(async () => {
    gatewayState().set('open')
  })

  expect(await screen.findByRole('button', { name: 'Launch crew' })).toBeTruthy()
  expect(screen.queryByRole('alert')).toBeNull()
})

it('still reports a list failure on an open socket', async () => {
  gatewayState().set('open')
  requestProfile.mockRejectedValue(new Error('groups.list exploded'))

  render(<CanonicalGroupList onOpen={() => undefined} />)

  expect((await screen.findByRole('alert')).textContent).toContain('groups.list exploded')
})
