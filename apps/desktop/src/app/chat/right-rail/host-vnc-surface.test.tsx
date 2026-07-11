import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getHostDisplay } from '@/hermes'
import { $activeGatewayProfile } from '@/store/profile'

import { HostVncSurface } from './host-vnc-surface'

vi.mock('@/hermes', () => ({
  getHostDisplay: vi.fn(),
  setApiRequestProfile: vi.fn()
}))

const getHostDisplayMock = vi.mocked(getHostDisplay)

describe('HostVncSurface', () => {
  afterEach(() => {
    cleanup()
    $activeGatewayProfile.set('default')
    vi.clearAllMocks()
  })

  it('embeds the configured noVNC page in a sandboxed peer surface', async () => {
    getHostDisplayMock.mockResolvedValue({
      available: true,
      reason: null,
      url: 'https://agent-host.example/vnc.html?autoconnect=true'
    })

    render(<HostVncSurface />)

    const frame = await screen.findByTitle('Host VNC')
    expect(frame.getAttribute('src')).toBe('https://agent-host.example/vnc.html?autoconnect=true')
    expect(frame.getAttribute('sandbox')).toBe('allow-forms allow-same-origin allow-scripts')
    expect(frame.getAttribute('referrerpolicy')).toBe('no-referrer')
  })

  it('rediscovers the host display when the active gateway profile changes', async () => {
    getHostDisplayMock.mockResolvedValue({
      available: true,
      reason: null,
      url: 'https://agent-host.example/vnc.html'
    })

    render(<HostVncSurface />)
    await screen.findByTitle('Host VNC')

    $activeGatewayProfile.set('work')

    await waitFor(() => expect(getHostDisplayMock).toHaveBeenCalledTimes(2))
  })

  it('shows the backend configuration guidance when Host VNC is unavailable', async () => {
    getHostDisplayMock.mockResolvedValue({
      available: false,
      reason: 'Set desktop.host_vnc_url to the host noVNC page',
      url: null
    })

    render(<HostVncSurface />)

    expect(await screen.findByText('Set desktop.host_vnc_url to the host noVNC page')).toBeDefined()
    expect(screen.queryByTitle('Host VNC')).toBeNull()
  })

  it('refuses to embed noVNC on the Hermes Webapp origin', async () => {
    getHostDisplayMock.mockResolvedValue({
      available: true,
      reason: null,
      url: `${window.location.origin}/vnc.html`
    })

    render(<HostVncSurface />)

    expect(
      await screen.findByText('Host VNC must use a dedicated origin or port, not the Hermes Webapp origin.')
    ).toBeDefined()
    expect(screen.queryByTitle('Host VNC')).toBeNull()
  })

  it('can retry discovery after a request failure', async () => {
    getHostDisplayMock.mockRejectedValueOnce(new Error('backend unavailable')).mockResolvedValueOnce({
      available: true,
      reason: null,
      url: 'http://127.0.0.1:6080/vnc.html'
    })

    render(<HostVncSurface />)

    expect(await screen.findByText('backend unavailable')).toBeDefined()
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))

    await waitFor(() => expect(getHostDisplayMock).toHaveBeenCalledTimes(2))
    expect(await screen.findByTitle('Host VNC')).toBeDefined()
  })
})
