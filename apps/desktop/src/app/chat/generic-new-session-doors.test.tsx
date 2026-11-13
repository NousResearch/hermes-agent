/**
 * The generic "New session" doors — the sidebar row and ⌘N — open a plain
 * session, never one owned by whichever bot an earlier action pinned.
 *
 * A bot chat pins its exact route as the next new chat's owner
 * (`pinNewChatRoute`) and nothing unpinned it: selecting another bot only
 * moves the Bot Mode scope. ⌘N cleared the profile half of that pin and kept
 * the route, so `resolveNewChatOwnerRoute` handed the next session to the
 * PREVIOUS bot. The bot-aimed doors (strip "+", ⌘T, "New chat with this bot")
 * read the selected bot's route on their own and are not touched here.
 *
 * With no saved default-profile preference the session goes to the root
 * `default` profile on the active source. Falling back to the ambient profile
 * opened the session in whichever bot was selected last.
 */

import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $defaultProfileRoute } from '@/store/default-profile'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  pinNewChatRoute,
  resolveNewChatOwnerRoute
} from '@/store/profile'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'

import { useNewSessionDoors } from './use-new-session-doors'

// The registry connection the active socket serves (null = legacy primary).
const gatewayConnection = vi.hoisted(() => ({ id: 'local' as null | string }))

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  activeGatewayConnectionId: () => gatewayConnection.id
}))

vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn(async () => undefined),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

const BOT_A = { connectionId: 'local', profile: 'bot-a' }

function mountDoors() {
  const openNewSessionTile = vi.fn(async () => undefined)
  const startFreshSessionDraft = vi.fn()
  const selectSidebarItem = vi.fn()
  const { result } = renderHook(() => useNewSessionDoors(openNewSessionTile, startFreshSessionDraft, selectSidebarItem))

  return { doors: result.current, openNewSessionTile, selectSidebarItem, startFreshSessionDraft }
}

afterEach(() => {
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  $defaultProfileRoute.set(null)
  $activeGatewayProfile.set('default')
  $projectScope.set(ALL_PROJECTS)
  gatewayConnection.id = 'local'
})

describe('generic New session doors drop a bot pinned by an earlier chat', () => {
  it('⌘N starts a session that is not owned by the previous bot', () => {
    pinNewChatRoute(BOT_A)
    const { doors, startFreshSessionDraft } = mountDoors()

    doors.startNewSession()

    expect(startFreshSessionDraft).toHaveBeenCalledOnce()
    expect($newChatRoute.get()).toEqual({ connectionId: 'local', profile: 'default' })
    expect($newChatProfile.get()).toBe('default')
    expect(resolveNewChatOwnerRoute()?.profile).toBe('default')
  })

  it('the sidebar row does the same before it routes the click', () => {
    pinNewChatRoute(BOT_A)
    const { doors, selectSidebarItem } = mountDoors()

    doors.navigateSidebar({ action: 'new-session', id: 'new-session', label: 'New session' } as never)

    expect(selectSidebarItem).toHaveBeenCalledOnce()
    expect($newChatRoute.get()).toEqual({ connectionId: 'local', profile: 'default' })
    expect(resolveNewChatOwnerRoute()?.profile).toBe('default')
  })

  it('the default-profile preference still names the owner', () => {
    pinNewChatRoute(BOT_A)
    // A saved preference other than the root profile wins over the fallback.
    $defaultProfileRoute.set({ connectionId: 'local', profile: 'work' })
    const { doors } = mountDoors()

    doors.startNewSession()

    expect($newChatRoute.get()).toEqual({ connectionId: 'local', profile: 'work' })
  })

  it('without a saved preference ⌘N opens the root default profile, not the bot left ambient', () => {
    // Selecting a bot moves the active gateway onto it; that ambient profile
    // is not a "default" the user chose.
    $activeGatewayProfile.set('bot-b')
    gatewayConnection.id = 'remote-1'
    const { doors } = mountDoors()

    doors.startNewSession()

    expect($newChatRoute.get()).toEqual({ connectionId: 'remote-1', profile: 'default' })
    expect(resolveNewChatOwnerRoute()?.profile).toBe('default')
  })

  it('the sidebar row opens the root default profile too', () => {
    $activeGatewayProfile.set('bot-b')
    gatewayConnection.id = 'remote-1'
    const { doors } = mountDoors()

    doors.navigateSidebar({ action: 'new-session', id: 'new-session', label: 'New session' } as never)

    expect(resolveNewChatOwnerRoute()?.profile).toBe('default')
  })

  it('a legacy primary (no registry connection) pins the default profile door', () => {
    $activeGatewayProfile.set('bot-b')
    gatewayConnection.id = null
    const { doors } = mountDoors()

    doors.startNewSession()

    expect($newChatRoute.get()).toBeNull()
    expect($newChatProfile.get()).toBe('default')
  })

  it('a project scope keeps its own owner', () => {
    $activeGatewayProfile.set('bot-b')
    gatewayConnection.id = 'remote-1'
    $projectScope.set('project-1')
    const { doors } = mountDoors()

    doors.startNewSession()

    expect($newChatRoute.get()).toBeNull()
    expect($newChatProfile.get()).toBeNull()
  })

  it('other sidebar rows keep the pin', () => {
    pinNewChatRoute(BOT_A)
    const { doors } = mountDoors()

    doors.navigateSidebar({ action: 'route', id: 'skills', label: 'Skills', route: '/skills' } as never)

    expect($newChatRoute.get()).toMatchObject(BOT_A)
  })
})
