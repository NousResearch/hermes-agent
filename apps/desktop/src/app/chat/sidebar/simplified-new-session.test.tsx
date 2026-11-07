import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import * as profile from '@/store/profile'
import { $agentSidebarChoice } from '@/store/simplified-agent-mode'
import { requestSimplifiedNewSession } from '@/store/simplified-new-session'
import { makeSessionInfo } from '@/test/session-info'

import { SimplifiedNewSession } from './simplified-new-session'
import { rankNewSessionAgents } from './simplified-new-session-model'

beforeEach(() => {
  $agentSidebarChoice.set(true)
  $apiRequestScope.set({ connectionId: 'alpha', profile: 'research' })
  $connectionsRegistry.set(null)
  $fleetRoster.set(null)
  profile.$activeGatewayProfile.set('research')
  profile.$profiles.set([
    { name: 'default', is_default: true },
    { name: 'research', is_default: false }
  ] as typeof profile.$profiles.value)
  vi.spyOn(profile, 'newSessionInAgent').mockImplementation(() => {})
  vi.spyOn(profile, 'newSessionInProfile').mockImplementation(() => {})
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  $agentSidebarChoice.set(false)
  $apiRequestScope.set({ connectionId: null, profile: null })
  profile.$activeGatewayProfile.set('default')
  profile.$profiles.set([])
  $fleetRoster.set(null)
})

function openPicker() {
  fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' })
}

it('starts directly with the sole agent, without a picker', () => {
  profile.$profiles.set([{ name: 'research', is_default: false }] as typeof profile.$profiles.value)
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  fireEvent.click(screen.getByRole('button', { name: 'New session' }))
  expect(screen.queryByRole('listbox')).toBeNull()
  expect(profile.newSessionInAgent).toHaveBeenCalledWith({ connectionId: 'alpha', profile: 'research' })
})

it('cancels without creating anything and keeps identically named agents on different gateways distinct', () => {
  $fleetRoster.set({
    primaryConnectionId: 'alpha',
    sources: [],
    agents: [
      {
        connectionId: 'beta',
        profile: 'research',
        connectionLabel: 'Beta',
        connectionKind: 'remote',
        handle: 'beta::research'
      }
    ]
  } as NonNullable<typeof $fleetRoster.value>)
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  openPicker()
  expect(screen.getAllByRole('option')[0].textContent).toContain('research')
  fireEvent.keyDown(screen.getByRole('listbox'), { key: 'Escape' })
  expect(profile.newSessionInAgent).not.toHaveBeenCalled()
  openPicker()
  fireEvent.click(screen.getByRole('option', { name: /research · Beta/ }))
  expect(profile.newSessionInAgent).not.toHaveBeenCalled()
  expect(screen.getByRole('combobox').getAttribute('aria-label')).toContain('research · Beta')
  fireEvent.click(screen.getByRole('button', { name: 'New session' }))
  expect(profile.newSessionInAgent).toHaveBeenCalledWith({ connectionId: 'beta', profile: 'research' })
})

it('starts blank with this gateway’s default agent and makes no automatic routing promise', () => {
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  openPicker()
  fireEvent.click(screen.getByRole('option', { name: /Start blank session/ }))
  expect(profile.newSessionInAgent).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'New session' }))
  expect(profile.newSessionInAgent).toHaveBeenCalledWith({ connectionId: 'alpha', profile: 'default' })
})

it('claims the shortcut only while the toggle is on and the picker is mounted', () => {
  expect(requestSimplifiedNewSession()).toBe(false)
  const view = render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  act(() => {
    expect(requestSimplifiedNewSession()).toBe(true)
  })
  expect(screen.queryByRole('listbox')).toBeNull()
  expect(profile.newSessionInAgent).toHaveBeenCalledWith({ connectionId: 'alpha', profile: 'research' })
  act(() => $agentSidebarChoice.set(false))
  expect(requestSimplifiedNewSession()).toBe(false)
  view.unmount()
  $agentSidebarChoice.set(true)
  expect(requestSimplifiedNewSession()).toBe(false)
})

it('falls back to the existing action while the roster is empty', () => {
  profile.$profiles.set([])
  const fallback = vi.fn()
  render(<SimplifiedNewSession onFallback={fallback} sessions={[]} />)
  fireEvent.click(screen.getByRole('button', { name: 'New session' }))
  expect(fallback).toHaveBeenCalledOnce()
})

it('ranks the current owner first, then balances frequency and recency without mixing gateways', () => {
  const agents = ['current', 'frequent', 'recent', 'old'].map(name => ({
    id: JSON.stringify(['alpha', name]),
    connectionId: 'alpha',
    profile: name,
    label: name,
    gateway: 'Alpha'
  }))

  const sessions = [
    makeSessionInfo({ id: '1', profile: 'frequent', last_active: 10000000 }),
    makeSessionInfo({ id: '2', profile: 'frequent', last_active: 9999999 }),
    makeSessionInfo({ id: '3', profile: 'recent', last_active: 10000001 }),
    makeSessionInfo({ id: '4', profile: 'old', last_active: 1 }),
    makeSessionInfo({ id: '5', connection_id: 'beta', profile: 'old', last_active: 10000001 })
  ]

  expect(rankNewSessionAgents(agents, sessions, 'alpha', 'current', 10000001).map(a => a.profile)).toEqual([
    'current',
    'frequent',
    'recent',
    'old'
  ])
})

it('supports older gateways without a registry connection identity', () => {
  $apiRequestScope.set({ connectionId: null, profile: 'research' })
  profile.$profiles.set([{ name: 'research', is_default: false }] as typeof profile.$profiles.value)
  render(
    <SimplifiedNewSession
      onFallback={vi.fn()}
      sessions={[makeSessionInfo({ id: 'orphan', profile: 'deleted-agent' })]}
    />
  )
  fireEvent.click(screen.getByRole('button', { name: 'New session' }))
  expect(profile.newSessionInProfile).toHaveBeenCalledWith('research')
  expect(profile.newSessionInAgent).not.toHaveBeenCalled()
  expect(screen.queryByRole('listbox')).toBeNull()
})

it('starts in the current agent with multiple agents without opening the picker', () => {
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  fireEvent.click(screen.getByRole('button', { name: 'New session' }))
  expect(screen.queryByRole('listbox')).toBeNull()
  expect(profile.newSessionInAgent).toHaveBeenCalledWith({ connectionId: 'alpha', profile: 'research' })
})

it('uses the selected agent for the shortcut and forgets that choice after changing gateway', () => {
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  openPicker()
  fireEvent.click(screen.getByRole('option', { name: /^default ·/ }))
  act(() => {
    requestSimplifiedNewSession()
  })
  expect(profile.newSessionInAgent).toHaveBeenLastCalledWith({ connectionId: 'alpha', profile: 'default' })
  act(() => $apiRequestScope.set({ connectionId: 'beta', profile: 'research' }))
  fireEvent.click(screen.getByRole('button', { name: 'New session' }))
  expect(profile.newSessionInAgent).toHaveBeenLastCalledWith({ connectionId: 'beta', profile: 'research' })
})

it('falls back to the current agent if the selected agent disappears', () => {
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  openPicker()
  fireEvent.click(screen.getByRole('option', { name: /^default ·/ }))
  act(() => profile.$profiles.set([{ name: 'research', is_default: false }] as typeof profile.$profiles.value))
  fireEvent.click(screen.getByRole('button', { name: 'New session' }))
  expect(profile.newSessionInAgent).toHaveBeenLastCalledWith({ connectionId: 'alpha', profile: 'research' })
})
