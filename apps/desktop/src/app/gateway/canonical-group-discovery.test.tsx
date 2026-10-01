import { gatewayActivationEpoch, host } from '@hermes/plugin-sdk'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $canonicalGroupBindings, $canonicalGroupNames, CanonicalGroupList } from '@/plugins/hermes-bots/canonical-group-registry'
import { readGroupExecutionMode } from '@/plugins/hermes-bots/canonical-groups'
import { CANONICAL_GROUP_CAPABILITIES } from '@/plugins/hermes-bots/group-test-utils'
import {
  configureGatewayRegistry,
  reportPrimaryGatewayState,
  setPrimaryGateway,
  setPrimaryGatewayConnection
} from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { setConnection } from '@/store/session'

vi.mock('@/plugins/hermes-bots/canonical-group-labels', () => ({
  useCanonicalGroupLabels: () => ({ refreshGroups: 'Refresh gateway groups' })
}))

const request = vi.fn()
let connectionSequence = 0
let connectionId: string

const gateway = {
  connectionState: 'connecting',
  request: async (method: string, params: Record<string, unknown>) => {
    if (gateway.connectionState !== 'open') {throw new Error('Hermes gateway unavailable')}

    return request(method, params)
  }
}

function changeSocket(state: 'connecting' | 'open' | 'closed') {
  gateway.connectionState = state
  reportPrimaryGatewayState(state)
}

beforeEach(() => {
  connectionId = `discovery-${++connectionSequence}`
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway(gateway as never, 'default')
  setPrimaryGatewayConnection({ connectionId, mode: 'local' })
  setConnection({ connectionId, mode: 'local', profile: 'default', port: 12345 } as never)
  $activeGatewayProfile.set('default')
  changeSocket('connecting')
  $canonicalGroupBindings.set({})
  $canonicalGroupNames.set({})
  request.mockReset()
})

afterEach(() => {
  cleanup()
  setPrimaryGateway(null)
  setConnection(null)
  reportPrimaryGatewayState('closed')
})

it('discovers retained rooms when the actual primary socket opens after the list mounts', async () => {
  const route = { connectionId, profile: 'default' }
  // An earlier surface can have classified the not-yet-open primary as
  // unavailable. Socket readiness does not advance the activation epoch.
  const epoch = gatewayActivationEpoch()
  expect((await readGroupExecutionMode(route, epoch)).mode).toBe('unavailable')
  expect(request).not.toHaveBeenCalled()
  request.mockImplementation(async method => method === 'groups.capabilities'
    ? CANONICAL_GROUP_CAPABILITIES
    : { rooms: [{ room_id: 'retained', name: 'Retained room', members: [] }], next_offset: null })

  render(<CanonicalGroupList onOpen={vi.fn()} />)
  await act(async () => {})
  expect(host.state.gateway.get()).toBe('connecting')
  expect(request).not.toHaveBeenCalled()

  await act(async () => { changeSocket('open') })
  await screen.findByRole('button', { name: 'Retained room' })
  expect(gatewayActivationEpoch()).toBe(epoch)
  expect(request.mock.calls).toEqual([
    ['groups.capabilities', { profile: 'default' }],
    ['groups.list', { limit: 100, offset: 0, profile: 'default' }]
  ])
  expect(Object.values($canonicalGroupBindings.get())).toEqual([{ ...route, roomId: 'retained' }])
})

it.each([false, true])('refreshes discovery after reconnect and refuses the old socket response (batched: %s)', async batched => {
  let releaseOld!: (page: unknown) => void
  let lists = 0
  request.mockImplementation(async method => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    if (++lists === 1) {return new Promise(resolve => { releaseOld = resolve })}

    return { rooms: [{ room_id: 'current', name: 'Current room', members: [] }], next_offset: null }
  })
  changeSocket('open')
  render(<CanonicalGroupList onOpen={vi.fn()} />)
  await waitFor(() => expect(lists).toBe(1))
  const epoch = gatewayActivationEpoch()
  const oldPage = { rooms: [{ room_id: 'stale', name: 'Old socket room', members: [] }], next_offset: null }

  if (batched) {
    await act(async () => {
      changeSocket('closed')
      changeSocket('open')
      // Return the old request before React has committed the batched
      // reconnect, so an effect-cleanup flag alone cannot fence it.
      releaseOld(oldPage)
    })
  } else {
    await act(async () => { changeSocket('closed') })
    await act(async () => { changeSocket('open') })
    await screen.findByRole('button', { name: 'Current room' })
    await act(async () => { releaseOld(oldPage) })
  }
  await screen.findByRole('button', { name: 'Current room' })

  expect(gatewayActivationEpoch()).toBe(epoch)
  expect(screen.queryByRole('button', { name: 'Old socket room' })).toBeNull()
  expect(Object.values($canonicalGroupBindings.get()).map(binding => binding.roomId)).toEqual(['current'])
  expect(request.mock.calls.filter(call => call[0] === 'groups.capabilities')).toHaveLength(2)
})
