// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CapabilityScope } from '../scope-selector'
import { CapabilityScopeSelector } from '../scope-selector'

const longProfile = 'research-program-with-a-deliberately-long-profile-name'

const compactScope: CapabilityScope = {
  crossBackend: false,
  key: 'default',
  onChange: vi.fn(),
  options: [
    { key: 'default', label: 'Hermes (default)', value: 'default' },
    { key: longProfile, label: longProfile, value: longProfile }
  ],
  profile: null,
  value: 'default'
}

describe('Plugins Agent scope selector', () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn()
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('preserves the exact selected profile through the compact value wrapper', async () => {
    render(<CapabilityScopeSelector compact scope={compactScope} />)

    const trigger = screen.getByRole('combobox')
    const value = trigger.querySelector<HTMLElement>('[data-slot="compact-select-value"]')
    expect(value?.textContent).toBe(compactScope.options[0].label)

    await act(async () => {
      fireEvent.click(trigger)
    })

    await screen.findByRole('listbox')
    const option = await screen.findByRole('option', { name: longProfile })
    fireEvent.click(option)
    expect(compactScope.onChange).toHaveBeenCalledWith(longProfile)
  })
})
