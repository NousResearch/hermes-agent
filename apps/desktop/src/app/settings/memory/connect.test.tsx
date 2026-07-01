import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const getMemoryProviderOAuthStatus = vi.fn()
const startMemoryProviderOAuth = vi.fn()

vi.mock('@/hermes', () => ({
  getMemoryProviderOAuthStatus: (provider: string) => getMemoryProviderOAuthStatus(provider),
  startMemoryProviderOAuth: (provider: string) => startMemoryProviderOAuth(provider)
}))

vi.mock('@/store/notifications', () => ({
  notifyError: vi.fn()
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('MemoryConnect', () => {
  it('does not probe OAuth status for OpenViking', async () => {
    const { MemoryConnect } = await import('./connect')
    const { container } = render(<MemoryConnect provider="openviking" />)

    await waitFor(() => expect(getMemoryProviderOAuthStatus).not.toHaveBeenCalled())
    expect(container.textContent).toBe('')
  })

  it('probes OAuth status for Honcho', async () => {
    getMemoryProviderOAuthStatus.mockResolvedValue({
      auth: 'oauth',
      connected: true,
      detail: '',
      state: 'connected'
    })

    const { MemoryConnect } = await import('./connect')
    render(<MemoryConnect provider="honcho" />)

    await waitFor(() => expect(getMemoryProviderOAuthStatus).toHaveBeenCalledWith('honcho'))
    expect(await screen.findByText('oauth set')).toBeTruthy()
  })
})
