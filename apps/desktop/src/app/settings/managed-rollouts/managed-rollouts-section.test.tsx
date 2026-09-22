import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { _resetManagedRolloutsForTests } from '@/store/managed-rollouts'
import { ManagedRolloutsSection } from './managed-rollouts-section'

describe('managed rollout section', () => {
  it('fails closed when the reviewed bridge is absent', () => {
    _resetManagedRolloutsForTests()
    render(<ManagedRolloutsSection history={[]} onSelect={() => undefined} />)
    expect(screen.queryByRole('region', { name: 'Managed rollouts' })).toBeNull()
  })
})
