import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { BROWSER_BRIDGE_STUBS } from '@/lib/browser-bridge-stubs'
import { $notifications, clearNotifications } from '@/store/notifications'

import { RemoteDisplayBanner } from './remote-display-banner'

function installReason(getRemoteDisplayReason?: () => Promise<string | null>) {
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { getRemoteDisplayReason }
}

beforeEach(() => {
  clearNotifications()
})

afterEach(() => {
  cleanup()
  clearNotifications()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('remote display banner', () => {
  it('stays silent in browser-hosted Desktop, where no GPU process was disabled', async () => {
    installReason(BROWSER_BRIDGE_STUBS.getRemoteDisplayReason)

    render(<RemoteDisplayBanner />)
    await act(async () => {})

    await expect(BROWSER_BRIDGE_STUBS.getRemoteDisplayReason()).resolves.toBeNull()
    expect($notifications.get()).toHaveLength(0)
  })

  it('reports the reason Electron actually disabled GPU acceleration for', async () => {
    const getRemoteDisplayReason = vi.fn(async () => 'rdp-session')
    installReason(getRemoteDisplayReason)

    render(<RemoteDisplayBanner />)

    await waitFor(() => expect($notifications.get()).toHaveLength(1))
    expect(getRemoteDisplayReason).toHaveBeenCalledTimes(1)
    expect($notifications.get()[0]).toMatchObject({ kind: 'info' })
    expect($notifications.get()[0].message).toContain('rdp-session')
    expect($notifications.get()[0].message).toContain('Software rendering active')
  })
})
