import './session-states'

import { beforeEach, expect, it, vi } from 'vitest'

// Reload the module to exercise startup hydration, not just the live atom.
beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

it('persists owner-scoped hide/show and layout through remounts and reloads', async () => {
  let prefs = await import('./agent-sidebar-preferences')
  const first = { connectionId: 'gateway-a', profile: 'research' }
  const second = { connectionId: 'gateway-b', profile: 'research' }
  prefs.setSidebarAgentHidden(first, true)
  prefs.setSidebarAgentHidden(first, true)
  prefs.$sidebarAgentExpanded.set({ [prefs.agentSidebarKey(first)]: false, [prefs.agentSidebarKey(second)]: true })
  prefs.$sidebarAgentScroll.set(427)
  prefs.$hiddenSidebarAgentsExpanded.set(true)
  const off = prefs.$hiddenSidebarAgents.subscribe(() => {})
  off()
  expect(prefs.$hiddenSidebarAgents.get()).toEqual([prefs.agentSidebarKey(first)])

  vi.resetModules()
  prefs = await import('./agent-sidebar-preferences')
  expect(prefs.$hiddenSidebarAgents.get()).toEqual([prefs.agentSidebarKey(first)])
  expect(prefs.$sidebarAgentExpanded.get()).toEqual({
    [prefs.agentSidebarKey(first)]: false,
    [prefs.agentSidebarKey(second)]: true
  })
  expect(prefs.$sidebarAgentScroll.get()).toBe(427)
  expect(prefs.$hiddenSidebarAgentsExpanded.get()).toBe(true)
  prefs.setSidebarAgentHidden(first, false)
  vi.resetModules()
  prefs = await import('./agent-sidebar-preferences')
  expect(prefs.$hiddenSidebarAgents.get()).toEqual([])
  expect(prefs.$sidebarAgentExpanded.get()[prefs.agentSidebarKey(first)]).toBe(false)
})

it('moves local preferences on profile rename and deletes only the exact target owner', async () => {
  const prefs = await import('./agent-sidebar-preferences')
  const local = { connectionId: null, profile: 'research' }
  const remote = { connectionId: 'remote', profile: 'research' }
  prefs.setSidebarAgentHidden(local, true)
  prefs.setSidebarAgentHidden(remote, true)
  prefs.$sidebarAgentExpanded.set({ [prefs.agentSidebarKey(local)]: false, [prefs.agentSidebarKey(remote)]: true })
  const { migrateTilesForProfile, dropTilesForProfile } = await import('./session-states')
  migrateTilesForProfile('research', 'renamed')
  const renamed = prefs.agentSidebarKey({ connectionId: 'local', profile: 'renamed' })
  expect(prefs.$hiddenSidebarAgents.get()).toEqual([renamed, prefs.agentSidebarKey(remote)])
  expect(prefs.$sidebarAgentExpanded.get()).toEqual({ [renamed]: false, [prefs.agentSidebarKey(remote)]: true })

  dropTilesForProfile('display-name', { connectionId: 'remote', profile: 'display-name', targetProfile: 'research' })
  expect(prefs.$hiddenSidebarAgents.get()).toEqual([renamed])
  expect(prefs.$sidebarAgentExpanded.get()).toEqual({ [renamed]: false })
  prefs.migrateAgentSidebarPreferences({ connectionId: 'local', profile: 'renamed' }, 'final')
  expect(prefs.$hiddenSidebarAgents.get()).toEqual([prefs.agentSidebarKey({ connectionId: null, profile: 'final' })])
})
