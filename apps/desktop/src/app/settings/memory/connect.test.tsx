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
  it('hides the connect affordance when the provider rejects the capability probe', async () => {
    getMemoryProviderOAuthStatus.mockRejectedValue(new Error('not supported'))

    const { MemoryConnect } = await import('./connect')
    const { container } = render(<MemoryConnect provider="openviking" />)

    await waitFor(() => expect(getMemoryProviderOAuthStatus).toHaveBeenCalledWith('openviking'))
    expect(container.textContent).toBe('')
  })

  it('renders any provider that reports OAuth capability', async () => {
    getMemoryProviderOAuthStatus.mockResolvedValue({
      auth: 'oauth',
      connected: true,
      detail: '',
      state: 'connected'
    })

    const { MemoryConnect } = await import('./connect')
    render(<MemoryConnect provider="future-provider" />)

    await waitFor(() => expect(getMemoryProviderOAuthStatus).toHaveBeenCalledWith('future-provider'))
    expect(await screen.findByText('oauth set')).toBeTruthy()
  })
})
