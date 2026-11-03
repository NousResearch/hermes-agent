/**
 * The two room-membership entry points offer only what the room can seat:
 * Manage members uses New Group Chat's seatable predicate for add candidates,
 * and New Group Chat is not offered when every seatable bot is on ONE other
 * connection (each selection there is refused), saying why instead.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $lastRoster } from './data'
import { GroupMemberPicker } from './group-chat-view-members'
import { botsText } from './i18n'
import { translateBots } from './i18n-test-helper'
import { renderRosterToolbar } from './roster-pane-toolbar'
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

it('Manage members will not uncheck a member whose machine is offline, and says why', async () => {
  const homelab: RosterRow = { ...remote('scout'), connectionLabel: 'Homelab', sourceReachable: false }
  $lastRoster.set([alpha, beta, homelab])
  const { baseElement } = render(<GroupMemberPicker group="Core" members={[alpha, beta, homelab]} onClose={vi.fn()} open />)

  await waitFor(() => expect(request.mock.calls.map(call => call[1])).toContain('groups.capabilities'))
  await act(async () => {})

  const row = [...baseElement.querySelectorAll('[data-testid="group-member-picker"] [role="checkbox"]')].find(node =>
    node.textContent?.includes('@scout')
  ) as HTMLElement

  expect(row.textContent).toContain(botsText().crossGateway.memberOffline('Homelab', 'Scout'))
  expect(row.hasAttribute('disabled')).toBe(true)
  fireEvent.click(row)
  expect(row.getAttribute('aria-checked')).toBe('true')
})

function renderToolbar(roster: RosterRow[]) {
  const setGroupCreateOpen = vi.fn()

  render(
    renderRosterToolbar({
      activeFilterCount: 0,
      activeSourceRoster: roster,
      activityFilter: 'all',
      activityToasts: false,
      b: botsText(),
      gatewayFilter: 'all',
      gatewayOptions: [],
      query: '',
      roster,
      rowKindFilter: 'all',
      setActivityFilter: vi.fn(),
      setCreateOpen: vi.fn(),
      setGatewayFilter: vi.fn(),
      setGroupCreateOpen,
      setQuery: vi.fn(),
      setRowKindFilter: vi.fn(),
      setSectionDialog: vi.fn(),
      showRosterFilters: false,
      showRosterSearch: false,
      showRosterTools: false
    })
  )
  fireEvent.keyDown(screen.getByRole('button', { name: botsText().roster.newBotOrGroup }), { key: 'Enter' })

  return setGroupCreateOpen
}

it('New Group Chat is disabled, with the reason, when every seatable bot is on one other connection', async () => {
  // This connection's only bot is one the authority will not seat.
  const setGroupCreateOpen = renderToolbar([{ name: 'parked', connectionId: 'local' }, remote('scout'), remote('scribe')])
  const note = botsText().crossGateway.selectionElsewhere('Other box')

  const item = await screen.findByRole('menuitem', { name: new RegExp(botsText().group.newTitle) })
  await waitFor(() => expect(item.hasAttribute('data-disabled')).toBe(true))
  expect(item.textContent).toContain(note)
  fireEvent.click(item)
  expect(setGroupCreateOpen).not.toHaveBeenCalled()
})

it('New Group Chat stays offered for a selection that can span connections', async () => {
  const setGroupCreateOpen = renderToolbar([alpha, remote('scout'), remote('scribe')])

  const item = await screen.findByRole('menuitem', { name: botsText().group.newTitle })
  await waitFor(() => expect(request.mock.calls.map(call => call[1])).toContain('groups.capabilities'))
  await act(async () => {})
  expect(item.hasAttribute('data-disabled')).toBe(false)
  fireEvent.click(item)
  expect(setGroupCreateOpen).toHaveBeenCalledOnce()
})
