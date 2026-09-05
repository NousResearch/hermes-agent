import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getConnection: vi.fn(),
  getGatewayWsUrl: vi.fn(),
  requestGateway: vi.fn(),
  toDataURL: vi.fn()
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: mocks.requestGateway })
}))

vi.mock('qrcode', () => ({
  default: { toDataURL: mocks.toDataURL }
}))

import { buildMobilePairingPayload, MobileCompanionPairing, readPublicGatewayUrl } from './mobile-companion-pairing'

beforeEach(() => {
  mocks.requestGateway.mockResolvedValue({
    config: { dashboard: { public_url: 'https://desktop.example.ts.net:9443' } }
  })
  mocks.getConnection.mockResolvedValue({
    authMode: 'token',
    profile: null,
    wsUrl: 'ws://127.0.0.1:51732/api/ws?token=stale'
  })
  mocks.getGatewayWsUrl.mockResolvedValue({
    ok: true,
    wsUrl: 'ws://127.0.0.1:51732/api/ws?token=fresh-secret'
  })
  mocks.toDataURL.mockResolvedValue('data:image/png;base64,pairing-code')

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      getConnection: mocks.getConnection,
      getGatewayWsUrl: mocks.getGatewayWsUrl
    }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('mobile companion pairing payload', () => {
  it('encodes the configured public gateway and fresh local token', () => {
    const payload = buildMobilePairingPayload(
      'https://desktop.example.ts.net/hermes/',
      'ws://127.0.0.1:51732/api/ws?token=fresh%20token'
    )

    const url = new URL(payload)

    expect(url.protocol).toBe('hermes-agent:')
    expect(url.hostname).toBe('desktop-pair')
    expect(url.searchParams.get('server')).toBe('https://desktop.example.ts.net/hermes')
    expect(url.searchParams.get('token')).toBe('fresh token')
  })

  it.each([
    'http://desktop.example.ts.net',
    'https://localhost:9443',
    'https://127.0.0.1:9443',
    'https://user:password@desktop.example.ts.net'
  ])('rejects an unsafe public gateway URL: %s', publicUrl => {
    expect(() => buildMobilePairingPayload(publicUrl, 'ws://127.0.0.1/api/ws?token=safe')).toThrow(
      'invalid-public-url'
    )
  })

  it('rejects one-time OAuth tickets and missing tokens', () => {
    expect(() =>
      buildMobilePairingPayload('https://desktop.example.ts.net', 'wss://desktop.example.ts.net/api/ws?ticket=once')
    ).toThrow('missing-token')
    expect(() => buildMobilePairingPayload('https://desktop.example.ts.net', 'ws://127.0.0.1/api/ws')).toThrow(
      'missing-token'
    )
  })

  it('reads only dashboard.public_url from the gateway config response', () => {
    expect(readPublicGatewayUrl({ config: { dashboard: { public_url: ' https://desktop.example.ts.net ' } } })).toBe(
      'https://desktop.example.ts.net'
    )
    expect(readPublicGatewayUrl({ config: { dashboard: {} } })).toBeNull()
    expect(readPublicGatewayUrl({ dashboard: { public_url: 'https://wrong-shape.example' } })).toBeNull()
  })

  it('reveals a QR only after a user asks and uses a freshly minted gateway token', async () => {
    render(<MobileCompanionPairing />)

    expect(screen.queryByRole('img')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Show pairing code' }))

    const qr = await screen.findByRole('img', { name: 'Hermes mobile companion pairing QR code' })

    expect(qr.getAttribute('src')).toBe('data:image/png;base64,pairing-code')
    expect(mocks.requestGateway).toHaveBeenCalledWith('config.get', { key: 'full' })
    expect(mocks.getGatewayWsUrl).toHaveBeenCalledWith(null)
    expect(mocks.toDataURL).toHaveBeenCalledTimes(1)

    const payload = mocks.toDataURL.mock.calls[0][0] as string

    expect(payload).toContain('server=https%3A%2F%2Fdesktop.example.ts.net%3A9443')
    expect(payload).toContain('token=fresh-secret')
    expect(screen.queryByText(/fresh-secret/)).toBeNull()
  })

  it('shows an actionable state instead of a broken QR when no public gateway URL is configured', async () => {
    mocks.requestGateway.mockResolvedValue({ config: { dashboard: {} } })
    render(<MobileCompanionPairing />)

    fireEvent.click(screen.getByRole('button', { name: 'Show pairing code' }))

    await waitFor(() => expect(screen.getByText(/Set dashboard\.public_url/)).toBeTruthy())
    expect(screen.queryByRole('img')).toBeNull()
    expect(mocks.toDataURL).not.toHaveBeenCalled()
  })
})
