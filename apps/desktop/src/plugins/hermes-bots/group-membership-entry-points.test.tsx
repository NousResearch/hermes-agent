/**
 * The room-membership entry points offer only what the room can seat: Manage
 * members uses New Group Chat's seatable predicate for add candidates.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $lastRoster } from './data'
import { GroupMemberPicker } from './group-chat-view-members'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const { request } = vi.hoisted(() => ({ request: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')

  return {
    ...sdk,
    gatewayActivationEpoch: () => 1,
    useGatewayActivationEpoch: () => 1,
    host: {
      ...sdk.host,
      requestProfile: request,
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

const remote = (name: string): RosterRow => ({
  name,
  connectionId: 'other-box',
  connectionLabel: 'Other box',
  remoteSource: true,
  route: { connectionId: 'other-box', mode: 'remote', profile: name, targetProfile: name }
})

// A row on this connection whose route names another one: New Group Chat hides it.
const spoofed: RosterRow = {
  name: 'spoofed',
  connectionId: 'local',
  remoteSource: true,
  route: { connectionId: 'other-box', mode: 'remote', profile: 'spoofed', targetProfile: 'spoofed' }
}

const alpha: RosterRow = { name: 'alpha', connectionId: 'local' }
const beta: RosterRow = { name: 'beta', connectionId: 'local' }

beforeEach(() => {
  request.mockReset().mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true, seatable_profiles: ['alpha', 'beta'] }}
    throw new Error(`Unexpected RPC: ${method}`)
  })
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(() => {
  cleanup()
  $lastRoster.set([])
})

it('Manage members offers the seatable rows plus the current members, never a route-disagreeing row', async () => {
  $lastRoster.set([alpha, beta, remote('scout'), spoofed])
  const { baseElement } = render(<GroupMemberPicker group="Core" members={[alpha, remote('scout')]} onClose={vi.fn()} open />)

  await waitFor(() => expect(request.mock.calls.map(call => call[1])).toContain('groups.capabilities'))
  await act(async () => {})

  const offered = [...baseElement.querySelectorAll('[data-testid="group-member-picker"] [role="checkbox"]')]
    .map(node => node.textContent || '')
    .join('\n')

  expect(offered).toContain('@alpha')
  expect(offered).toContain('@beta')
  expect(offered).toContain('Other box')
  expect(offered).not.toContain('spoofed')
})
