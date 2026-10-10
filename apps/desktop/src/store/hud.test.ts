import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { $hudActive, $hudSession, openHud } from './hud'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const open = vi.fn().mockResolvedValue({ ok: true })

function installBridge() {
  desktopWindow.hermesDesktop = {
    hud: { open }
  } as unknown as Window['hermesDesktop']
}

function session(overrides: Partial<SessionInfo>): SessionInfo {
  return { id: 's', title: '', created_at: '', updated_at: '', ...overrides } as SessionInfo
}

beforeEach(() => {
  open.mockClear()
  installBridge()
  $hudActive.set(false)
  $hudSession.set(null)
  $sessions.set([])
  $activeGatewayProfile.set('default')
  $connection.set(null)
})

afterEach(() => {
  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('openHud profile targeting (#82285)', () => {
  it('carries the session-stamped profile when the target belongs to another profile', () => {
    $sessions.set([session({ id: 'abc', profile: 'work' })])
    $activeGatewayProfile.set('default')

    openHud('abc')

    expect(open).toHaveBeenCalledWith({ sessionId: 'abc', profile: 'work', connectionId: null })
  })

  it('falls back to the active gateway profile for an unstamped session', () => {
    $sessions.set([session({ id: 'abc', profile: '' })])
    $activeGatewayProfile.set('work')

    openHud('abc')

    expect(open).toHaveBeenCalledWith({ sessionId: 'abc', profile: 'work', connectionId: null })
  })

  it('uses the active gateway profile when opening without a session', () => {
    $activeGatewayProfile.set('research')

    openHud()

    expect(open).toHaveBeenCalledWith({ sessionId: null, profile: 'research', connectionId: null })
  })

  it('uses the active profile when the target session is not in the cache', () => {
    $activeGatewayProfile.set('work')

    openHud('unknown-session')

    expect(open).toHaveBeenCalledWith({ sessionId: 'unknown-session', profile: 'work', connectionId: null })
  })

  it('carries the connection this window is on, so a HUD from This device stays local', () => {
    $connection.set({ connectionId: 'local', profile: 'default' } as never)

    openHud()

    expect(open).toHaveBeenCalledWith({ sessionId: null, profile: 'default', connectionId: 'local' })
  })

  it("prefers the session's stamped owner connection over the window's", () => {
    $connection.set({ connectionId: 'local', profile: 'default' } as never)
    $sessions.set([session({ id: 'abc', profile: 'default', connection_id: 'studio' } as Partial<SessionInfo>)])

    openHud('abc')

    expect(open).toHaveBeenCalledWith({ sessionId: 'abc', profile: 'default', connectionId: 'studio' })
  })
})
