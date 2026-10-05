import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $desktopBoot } from '@/store/boot'
import { $desktopOnboarding } from '@/store/onboarding'
import { $connection } from '@/store/session'
import { $updateOverlayOpen, $updateOverlayTarget, setUpdateOverlayOpen } from '@/store/updates'

import { BootFailureOverlay } from './boot-failure-overlay'

const remoteToken = {
  envOverride: false,
  mode: 'remote',
  profile: null,
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: true,
  remoteUrl: 'http://unreachable.example.test:9191',
  cloudOrg: ''
}

beforeEach(() => {
  // Remote mode normally makes the generic updater choose the backend. Seed
  // that state so this test proves the recovery button overrides it.
  $connection.set({
    baseUrl: remoteToken.remoteUrl,
    isFullscreen: false,
    logs: [],
    mode: 'remote',
    nativeOverlayWidth: 0,
    token: 'test-token',
    windowButtonPosition: null,
    wsUrl: 'ws://unreachable.example.test:9191/ws'
  })
  $updateOverlayTarget.set('backend')
  $desktopOnboarding.set({
    configured: true,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    freeTierReady: false
  })
  $desktopBoot.set({
    error: 'Could not reach the remote Hermes gateway while refreshing its WebSocket ticket.',
    fakeMode: false,
    message: 'boot failed',
    phase: 'renderer.error',
    progress: 40,
    running: false,
    timestamp: Date.now(),
    visible: true
  })
  setUpdateOverlayOpen(false)
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  setUpdateOverlayOpen(false)
})

describe('BootFailureOverlay desktop update recovery', () => {
  it('opens the existing local client updater when the remote gateway is unreachable', async () => {
    const original = window.hermesDesktop
    const check = vi.fn().mockResolvedValue({ supported: true, behind: 0, fetchedAt: Date.now() })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        getRecentLogs: async () => ({ lines: [] }),
        getConnectionConfig: async () => remoteToken,
        getBootstrapState: async () => ({
          active: false,
          manifest: null,
          stages: {},
          error: null,
          log: [],
          startedAt: null,
          completedAt: null,
          setupChoice: null,
          unsupportedPlatform: null,
          bundled: false
        }),
        updates: { check }
      }
    })

    try {
      render(<BootFailureOverlay />)

      const update = await screen.findByRole('button', { name: /update app/i })
      fireEvent.click(update)

      expect($updateOverlayTarget.get()).toBe('client')
      expect($updateOverlayOpen.get()).toBe(true)
      expect(screen.queryByRole('dialog', { name: /Hermes couldn't start/i })).toBeNull()
      await waitFor(() => expect(check).toHaveBeenCalledWith({ force: true }))
    } finally {
      Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: original })
    }
  })
})
