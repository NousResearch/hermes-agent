import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { $hudActive, $hudSession, openHud, resetHudLayout, toggleHud } from './hud'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const open = vi.fn().mockResolvedValue({ ok: true })
const resetLayout = vi.fn().mockResolvedValue({ ok: true })
const close = vi.fn().mockResolvedValue({ ok: true })

function installBridge() {
  desktopWindow.hermesDesktop = {
    hud: { close, open, resetLayout }
  } as unknown as Window['hermesDesktop']
}

function session(overrides: Partial<SessionInfo>): SessionInfo {
  return { id: 's', title: '', created_at: '', updated_at: '', ...overrides } as SessionInfo
}

beforeEach(() => {
  open.mockClear()
  resetLayout.mockClear()
  close.mockClear()
  installBridge()
  $hudActive.set(false)
  $hudSession.set(null)
  $sessions.set([])
  $connection.set(null)
  $activeGatewayProfile.set('default')
})

afterEach(() => {
  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('resetHudLayout', () => {
  it('uses the native HUD recovery capability', () => {
    resetHudLayout()

    expect(resetLayout).toHaveBeenCalledOnce()
  })
})

describe('openHud profile targeting (#82285)', () => {
  it('carries the session-stamped profile when the target belongs to another profile', () => {
    $sessions.set([session({ id: 'abc', profile: 'work' })])
    $activeGatewayProfile.set('default')

    openHud('abc')

    expect(open).toHaveBeenCalledWith({ sessionId: 'abc', profile: 'work' })
  })

  it('carries the session owner connection when the target row is connection-tagged', () => {
    $sessions.set([session({ id: 'abc', profile: 'default', connection_id: 'alia-linux' })])

    openHud('abc')

    expect(open).toHaveBeenCalledWith({ sessionId: 'abc', profile: 'default', connectionId: 'alia-linux' })
  })

  it('falls back to the active gateway profile for an unstamped session', () => {
    $sessions.set([session({ id: 'abc', profile: '' })])
    $activeGatewayProfile.set('work')

    openHud('abc')

    expect(open).toHaveBeenCalledWith({ sessionId: 'abc', profile: 'work' })
  })

  it('uses the active gateway profile when opening without a session', () => {
    $activeGatewayProfile.set('research')

    openHud()

    expect(open).toHaveBeenCalledWith({ sessionId: null, profile: 'research' })
  })

  it('carries the active connection when opening without a session', () => {
    $connection.set({ mode: 'remote', connectionId: 'alia-linux' } as never)

    openHud()

    expect(open).toHaveBeenCalledWith({ sessionId: null, profile: 'default', connectionId: 'alia-linux' })
  })

  it('normalizes to default for single-profile users', () => {
    openHud()

    expect(open).toHaveBeenCalledWith({ sessionId: null, profile: 'default' })
  })

  it('uses the active profile when the target session is not in the cache', () => {
    $activeGatewayProfile.set('work')

    openHud('unknown-session')

    expect(open).toHaveBeenCalledWith({ sessionId: 'unknown-session', profile: 'work' })
  })
})

describe('toggleHud retargeting an open HUD', () => {
  it('points the open HUD at the conversation the toggle came from', () => {
    $hudActive.set(true)
    $hudSession.set('old')

    toggleHud('new')

    // Main owns the retarget (openHudWindow sends `hud:goto`), so reaching it
    // at all means going back through the open bridge — not closing the window.
    expect(open).toHaveBeenCalledWith({ sessionId: 'new', profile: 'default' })
    expect(close).not.toHaveBeenCalled()
  })

  it('carries the target session profile across a retarget', () => {
    $sessions.set([session({ id: 'new', profile: 'work' })])
    $hudActive.set(true)
    $hudSession.set('old')

    toggleHud('new')

    expect(open).toHaveBeenCalledWith({ sessionId: 'new', profile: 'work' })
  })

  it('dismisses when the HUD is already on that conversation', () => {
    $hudActive.set(true)
    $hudSession.set('same')

    toggleHud('same')

    expect(close).toHaveBeenCalled()
    expect(open).not.toHaveBeenCalled()
  })

  it('dismisses when there is no conversation to retarget onto', () => {
    $hudActive.set(true)
    $hudSession.set('old')

    toggleHud(null)

    expect(close).toHaveBeenCalled()
    expect(open).not.toHaveBeenCalled()
  })

  it('opens on the given conversation when the HUD is down', () => {
    $hudActive.set(false)

    toggleHud('new')

    expect(open).toHaveBeenCalledWith({ sessionId: 'new', profile: 'default' })
    expect(close).not.toHaveBeenCalled()
  })
})
