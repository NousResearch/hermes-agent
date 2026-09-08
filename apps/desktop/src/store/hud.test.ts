import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { $hudActive, $hudOrientation, $hudSession, openHud, resetHudLayout, setHudOrientation } from './hud'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const open = vi.fn().mockResolvedValue({ ok: true })
const resetLayout = vi.fn().mockResolvedValue({ ok: true })

function installBridge() {
  desktopWindow.hermesDesktop = {
    hud: { open, resetLayout }
  } as unknown as Window['hermesDesktop']
}

function session(overrides: Partial<SessionInfo>): SessionInfo {
  return { id: 's', title: '', created_at: '', updated_at: '', ...overrides } as SessionInfo
}

beforeEach(() => {
  open.mockClear()
  resetLayout.mockClear()
  installBridge()
  $hudActive.set(false)
  $hudSession.set(null)
  $sessions.set([])
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

describe('hud orientation', () => {
  const ORIENTATION_KEY = 'hermes.desktop.hud.orientation'

  afterEach(() => {
    window.localStorage.removeItem(ORIENTATION_KEY)
    setHudOrientation('composer-top')
  })

  it('boots on the shipped composer-top layout when nothing is stored', async () => {
    window.localStorage.removeItem(ORIENTATION_KEY)
    vi.resetModules()

    const { $hudOrientation: fresh } = await import('./hud')

    expect(fresh.get()).toBe('composer-top')
  })

  it('boots on the persisted choice, and falls back on junk', async () => {
    window.localStorage.setItem(ORIENTATION_KEY, JSON.stringify('composer-bottom'))
    vi.resetModules()

    const { $hudOrientation: restored } = await import('./hud')

    expect(restored.get()).toBe('composer-bottom')

    window.localStorage.setItem(ORIENTATION_KEY, '{corrupt')
    vi.resetModules()

    const { $hudOrientation: fallback } = await import('./hud')

    expect(fallback.get()).toBe('composer-top')
  })

  it('persists the choice so a reopened HUD restores it', () => {
    setHudOrientation('composer-bottom')

    expect($hudOrientation.get()).toBe('composer-bottom')
    expect(JSON.parse(window.localStorage.getItem(ORIENTATION_KEY) ?? '')).toBe('composer-bottom')
  })
})

describe('openHud profile targeting (#82285)', () => {
  it('carries the session-stamped profile when the target belongs to another profile', () => {
    $sessions.set([session({ id: 'abc', profile: 'work' })])
    $activeGatewayProfile.set('default')

    openHud('abc')

    expect(open).toHaveBeenCalledWith({ sessionId: 'abc', profile: 'work' })
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
