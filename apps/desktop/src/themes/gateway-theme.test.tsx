import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import { modePref, skinPref } from './appearance-prefs'
import { committedTheme, ThemeProvider, useTheme } from './context'

let theme: ReturnType<typeof useTheme>

function Probe() {
  theme = useTheme()

  return null
}

function hop(connectionId: string, profile = 'default', port = 10001) {
  act(() => {
    $activeGatewayProfile.set(profile)
    $connection.set({
      connectionId,
      profile,
      mode: connectionId === 'local' ? 'local' : 'remote',
      baseUrl: `http://127.0.0.1:${port}`,
      wsUrl: `ws://127.0.0.1:${port}/ws`,
      token: '',
      logs: [],
      isFullscreen: false,
      nativeOverlayWidth: 0,
      windowButtonPosition: null
    } satisfies HermesConnection)
  })
}

const mount = () =>
  render(
    <ThemeProvider>
      <Probe />
    </ThemeProvider>
  )

beforeEach(() => {
  window.localStorage.clear()
  hop('local')
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  $activeGatewayProfile.set('default')
})

describe('ThemeProvider gateway appearance', () => {
  it('persists an unidentified remote separately from local and registered gateways without tunnel keys', () => {
    mount()
    act(() => {
      theme.setTheme('catppuccin')
      theme.setMode('dark')
    })
    hop('sung-server')
    act(() => {
      theme.setTheme('nous')
      theme.setMode('system')
    })
    act(() => $connection.set({ ...$connection.get(), connectionId: undefined, mode: 'remote' } as HermesConnection))
    act(() => {
      theme.setTheme('ember')
      theme.setMode('light')
    })
    cleanup()
    mount()
    expect([theme.themeName, theme.mode]).toEqual(['ember', 'light'])
    hop('local')
    expect([theme.themeName, theme.mode]).toEqual(['catppuccin', 'dark'])
    hop('sung-server')
    expect([theme.themeName, theme.mode]).toEqual(['nous', 'system'])
    hop('sung-server', 'default', 30003)
    act(() => $connection.set({ ...$connection.get(), connectionId: undefined, mode: 'remote' } as HermesConnection))
    expect([theme.themeName, theme.mode]).toEqual(['ember', 'light'])

    for (const key of ['hermes-desktop-profile-themes-v2', 'hermes-desktop-profile-modes-v2']) {
      expect(window.localStorage.getItem(key)).not.toMatch(/127\.0\.0\.1|https?:|wss?:/)
    }
  })

  it('keeps the remote owner during a null reconnect descriptor', () => {
    mount()
    act(() => {
      theme.setTheme('catppuccin')
      theme.setMode('dark')
    })
    hop('sung-server')
    act(() => {
      theme.setTheme('nous')
      theme.setMode('light')
    })
    act(() => $connection.set(null))
    expect([theme.themeName, theme.mode]).toEqual(['nous', 'light'])
    act(() => {
      theme.setTheme('ember')
      theme.setMode('system')
    })
    hop('sung-server')
    expect([theme.themeName, theme.mode]).toEqual(['ember', 'system'])
    hop('local')
    expect([theme.themeName, theme.mode]).toEqual(['catppuccin', 'dark'])
  })

  it('keeps independent modes and profiles on each gateway', () => {
    mount()
    act(() => {
      theme.setTheme('catppuccin')
      theme.setMode('dark')
    })
    hop('sung-server')
    act(() => {
      theme.setTheme('nous')
      theme.setMode('light')
    })
    hop('local', 'work')
    act(() => {
      theme.setTheme('ember')
      theme.setMode('system')
    })
    hop('local')
    expect([theme.themeName, theme.mode]).toEqual(['catppuccin', 'dark'])
    hop('sung-server', 'work')
    act(() => {
      theme.setTheme('midnight')
      theme.setMode('dark')
    })
    hop('sung-server')
    expect([theme.themeName, theme.mode]).toEqual(['nous', 'light'])
    hop('local', 'work')
    expect([theme.themeName, theme.mode]).toEqual(['ember', 'system'])
    act(() => theme.setTheme('everforest'))
    expect(theme.mode).toBe('system')
    act(() => theme.setMode('light'))
    expect(theme.themeName).toBe('everforest')
    hop('local')
    expect([theme.themeName, theme.mode]).toEqual(['catppuccin', 'dark'])
  })

  it('does not carry a transient theme preview to another gateway', () => {
    mount()
    act(() => theme.setTheme('catppuccin'))
    hop('sung-server')
    act(() => theme.setTheme('nous'))
    act(() => theme.previewTheme('ember', 'dark'))
    hop('local')
    expect(theme.themeName).toBe('catppuccin')
    expect(window.document.documentElement.dataset.hermesTheme).toBe('catppuccin')
  })

  it('reloads both gateway picks and first-paints the remembered remote route', async () => {
    mount()
    act(() => {
      theme.setTheme('catppuccin')
      theme.setMode('dark')
    })
    hop('sung-server')
    act(() => {
      theme.setTheme('nous')
      theme.setMode('light')
    })
    cleanup()
    $connection.set(null)
    vi.resetModules()
    const fresh = await import('./context')
    const { $connection: freshConnection } = await import('@/store/session')
    expect(window.document.documentElement.dataset.hermesTheme).toBe('nous')
    expect(window.document.documentElement.dataset.hermesMode).toBe('light')

    function FreshProbe() {
      theme = fresh.useTheme()

      return null
    }

    render(
      <fresh.ThemeProvider>
        <FreshProbe />
      </fresh.ThemeProvider>
    )
    expect(theme.themeName).toBe('nous')
    expect(theme.mode).toBe('light')
    act(() =>
      freshConnection.set({ ...$connection.get(), connectionId: 'local', profile: 'default' } as HermesConnection)
    )
    expect(theme.themeName).toBe('catppuccin')
    expect(theme.mode).toBe('dark')
    act(() => freshConnection.set({ ...freshConnection.get(), connectionId: 'sung-server' } as HermesConnection))
    expect(theme.themeName).toBe('nous')
    expect(theme.mode).toBe('light')
    cleanup()
    freshConnection.set(null)
    vi.resetModules()
  })

  it('uses the same gateway route for committed accent reads and provider writes', () => {
    mount()
    act(() => theme.setTheme('catppuccin'))
    hop('sung-server', 'research')
    // The descriptor is authoritative even while the ambient profile lags a hop.
    act(() => $activeGatewayProfile.set('default'))
    act(() => theme.setTheme('ember'))
    expect(theme.themeName).toBe('ember')
    expect(committedTheme().name).toBe('ember')
    expect(skinPref.resolve('research', 'sung-server')).toBe('ember')
    expect(skinPref.resolve('default', 'local')).toBe('catppuccin')
    hop('sung-server', 'research')
    expect(theme.themeName).toBe('ember')
  })

  it('syncs peer-window appearance only from the visible gateway', () => {
    mount()
    act(() => theme.setTheme('catppuccin'))
    hop('sung-server')
    act(() => theme.setTheme('nous'))
    // A different window writes local's preferences, then Chromium broadcasts storage.
    skinPref.assign('default', 'ember', 'local')
    modePref.assign('default', 'light', 'local')
    act(() => window.dispatchEvent(new StorageEvent('storage', { key: 'hermes-desktop-profile-themes-v2' })))
    expect(theme.themeName).toBe('nous')
    skinPref.assign('default', 'everforest', 'sung-server')
    modePref.assign('default', 'dark', 'sung-server')
    act(() => window.dispatchEvent(new StorageEvent('storage', { key: null })))
    expect(theme.themeName).toBe('everforest')
    expect(theme.mode).toBe('dark')
    expect(window.document.documentElement.dataset.hermesTheme).toBe('everforest')
  })

  it('restores local default after a same-name remote default picks another theme', () => {
    mount()
    act(() => theme.setTheme('catppuccin'))
    hop('sung-server')
    act(() => theme.setTheme('nous'))
    hop('local', 'default', 20002)
    expect(theme.themeName).toBe('catppuccin')
    expect(window.document.documentElement.dataset.hermesTheme).toBe('catppuccin')
    hop('sung-server', 'default', 30003)
    expect(theme.themeName).toBe('nous')
    expect(window.document.documentElement.dataset.hermesTheme).toBe('nous')
  })
})
