import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { _resetManagedRolloutsForTests } from '@/store/managed-rollouts'
import { ManagedRolloutsSection } from './managed-rollouts-section'

describe('managed rollout section', () => {
  afterEach(() => {
    _resetManagedRolloutsForTests()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: undefined })
  })

  it('fails closed when the reviewed bridge is absent', () => {
    render(<ManagedRolloutsSection history={[]} onSelect={() => undefined} />)
    expect(screen.queryByRole('region', { name: 'Managed rollouts' })).toBeNull()
  })

  it('uses the nested connections bridge and surfaces unsupported capability state', async () => {
    const capabilities = vi.fn().mockResolvedValue({
      protocol: 1,
      available: false,
      reason: 'trusted-assurance-provider-unavailable',
      maxConcurrency: 0,
      maxInstallations: 0
    })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { connections: { managedRollouts: { capabilities } } }
    })

    render(<ManagedRolloutsSection history={[]} onSelect={() => undefined} />)
    expect(screen.getByRole('region', { name: 'Managed rollouts' })).toBeTruthy()
    await waitFor(() => expect(capabilities).toHaveBeenCalledTimes(1))
    expect(screen.getByText('trusted-assurance-provider-unavailable')).toBeTruthy()
  })
})
