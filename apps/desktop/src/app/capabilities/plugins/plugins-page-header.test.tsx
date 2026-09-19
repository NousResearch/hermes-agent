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

  it('truncates the compact trigger and caps its popup to the viewport', async () => {
    render(<CapabilityScopeSelector compact scope={compactScope} />)

    const trigger = screen.getByRole('combobox')
    const value = trigger.querySelector<HTMLElement>('[data-slot="compact-select-value"]')
    expect(trigger.classList.contains('min-w-0')).toBe(true)
    expect(trigger.classList.contains('truncate')).toBe(true)
    expect(value?.classList.contains('min-w-0')).toBe(true)
    expect(value?.classList.contains('truncate')).toBe(true)

    await act(async () => {
      fireEvent.click(trigger)
    })

    const popup = await screen.findByRole('listbox')
    expect(popup.classList.contains('max-w-(--radix-select-available-width)')).toBe(true)
    expect(await screen.findByRole('option', { name: longProfile })).toBeTruthy()
  })
})
