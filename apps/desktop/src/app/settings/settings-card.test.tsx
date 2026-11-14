import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { ListRow, ToggleRow } from './primitives'
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

  it('counts the settings it renders when no count is given, and follows rows that mount later', async () => {
    act(() => $shellChoice.set(true))

    const { rerender } = render(
      <SettingsDisclosure title="Advanced">
        <ListRow title="Interface scale" />
        <ToggleRow checked={false} label="Thread timeline" onChange={() => {}} />
      </SettingsDisclosure>
    )

    const summary = () => screen.getByText('Advanced').closest('summary')!

    expect(summary().textContent).toContain('2')

    rerender(
      <SettingsDisclosure title="Advanced">
        <ListRow title="Interface scale" />
        <ToggleRow checked={false} label="Thread timeline" onChange={() => {}} />
        <ListRow below={<ListRow title="Nested detail" />} title="Chat font" />
      </SettingsDisclosure>
    )

    // A row inside another row's `below` is part of that setting, not a second one.
    await waitFor(() => expect(summary().textContent).toContain('3'))
  })

  it('shows no badge when it holds no settings', () => {
    act(() => $shellChoice.set(true))

    render(
      <SettingsDisclosure title="Advanced">
        <p>Build details</p>
      </SettingsDisclosure>
    )

    expect(
      screen.getByText('Advanced').closest('summary')!.querySelector('.global-settings-disclosure-count')
    ).toBeNull()
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
