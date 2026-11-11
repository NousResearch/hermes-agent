import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { SettingsCard, SettingsDisclosure } from './settings-card'

afterEach(() => {
  act(() => $shellChoice.set(null))
})

describe('SettingsDisclosure', () => {
  it('collapses its settings behind a summary that shows the count in the redesign', () => {
    act(() => $shellChoice.set(true))

    render(
      <SettingsDisclosure count={3} description="Rarely changed" title="Advanced">
        <button type="button">Hidden setting</button>
      </SettingsDisclosure>
    )

    const summary = screen.getByText('Advanced').closest('summary')!
    const details = summary.closest('details')!

    expect(details.open).toBe(false)
    expect(summary.textContent).toContain('3')
    expect(summary.textContent).toContain('Rarely changed')

    fireEvent.click(summary)

    expect(details.open).toBe(true)
  })

  it('keeps classic settings always visible under the classic heading', () => {
    act(() => $shellChoice.set(false))

    const { container } = render(
      <SettingsDisclosure classicHeading={<h3>Archived</h3>} count={3} title="Archived sessions">
        <button type="button">Restore</button>
      </SettingsDisclosure>
    )

    expect(container.querySelector('details')).toBeNull()
    expect(screen.getByRole('heading', { name: 'Archived' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Restore' })).toBeTruthy()
    expect(screen.queryByText('Archived sessions')).toBeNull()
  })
})

describe('SettingsCard', () => {
  it('passes rows straight through to the classic layout', () => {
    act(() => $shellChoice.set(false))

    const { container } = render(
      <SettingsCard classicClassName="grid gap-3" id="updates">
        <span>Row</span>
      </SettingsCard>
    )

    const card = container.querySelector('#updates')!

    expect(card.className).toBe('grid gap-3')
    expect(card.textContent).toBe('Row')
  })
})
