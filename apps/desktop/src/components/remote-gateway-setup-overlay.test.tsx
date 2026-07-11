// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RemoteGatewaySetupOverlay } from './remote-gateway-setup-overlay'

vi.mock('@/app/settings/gateway-settings', () => ({
  GatewaySettings: () => <div>Remote gateway setup</div>
}))

afterEach(() => {
  cleanup()
})

describe('RemoteGatewaySetupOverlay', () => {
  it('opens for an unconfigured remote-only build', async () => {
    window.hermesDesktop = {
      getConnectionConfig: vi.fn(async () => ({ mode: 'local' })),
      getDesktopCapabilities: vi.fn(async () => ({ remoteOnly: true }))
    } as unknown as Window['hermesDesktop']

    render(<RemoteGatewaySetupOverlay />)

    expect(await screen.findByText('Remote gateway setup')).toBeTruthy()
  })

  it('stays closed for configured remote clients', async () => {
    window.hermesDesktop = {
      getConnectionConfig: vi.fn(async () => ({ mode: 'remote' })),
      getDesktopCapabilities: vi.fn(async () => ({ remoteOnly: true }))
    } as unknown as Window['hermesDesktop']

    render(<RemoteGatewaySetupOverlay />)

    await waitFor(() => expect(window.hermesDesktop.getConnectionConfig).toHaveBeenCalled())
    expect(screen.queryByText('Remote gateway setup')).toBeNull()
  })

  it('stays closed for the full desktop app', async () => {
    window.hermesDesktop = {
      getConnectionConfig: vi.fn(async () => ({ mode: 'local' })),
      getDesktopCapabilities: vi.fn(async () => ({ remoteOnly: false }))
    } as unknown as Window['hermesDesktop']

    render(<RemoteGatewaySetupOverlay />)

    await waitFor(() => expect(window.hermesDesktop.getDesktopCapabilities).toHaveBeenCalled())
    expect(screen.queryByText('Remote gateway setup')).toBeNull()
  })
})
