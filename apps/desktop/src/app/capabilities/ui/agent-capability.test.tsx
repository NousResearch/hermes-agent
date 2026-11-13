import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { Skeleton } from '@/components/ui/skeleton'

import { AgentCapabilityBanner } from './agent-capability'

afterEach(cleanup)

it('holds a loading count line in the banner without nesting a block inside a paragraph', () => {
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {})

  try {
    const { container, getByText } = render(
      <AgentCapabilityBanner action={null} description={<Skeleton />} title="Give this agent a skill" />
    )

    expect(getByText('Give this agent a skill')).toBeTruthy()
    expect(container.querySelector('[data-slot="skeleton"]')).not.toBeNull()
    expect(container.querySelector('p div, p [data-slot="skeleton"]')).toBeNull()
    expect(errors.mock.calls.flat().join(' ')).not.toMatch(/cannot be a descendant|cannot contain a nested/)
  } finally {
    errors.mockRestore()
  }
})
