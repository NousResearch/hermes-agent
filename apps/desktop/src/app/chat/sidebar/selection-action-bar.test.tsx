import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $sidebarSelection, clearSidebarSelection } from '@/store/sidebar-selection'

import { SelectionActionBar } from './selection-action-bar'

afterEach(() => {
  cleanup()
  clearSidebarSelection()
})

describe('SelectionActionBar', () => {
  it('keeps the selected count readable while actions take the remaining width', () => {
    $sidebarSelection.set({ ids: ['s1', 's2'], section: 'sessions' })

    const { container } = render(
      <I18nProvider configClient={null}>
        <SelectionActionBar sessions={[]} />
      </I18nProvider>
    )

    const label = screen.getByText('2 selected')
    expect(label.hasAttribute('data-selection-count-label')).toBe(true)
    expect(label.className).toContain('whitespace-nowrap')
    expect(label.className).not.toContain('truncate')

    const count = container.querySelector('[data-selection-count]') as HTMLElement
    expect(count.className).toContain('shrink-0')
    expect(count.className).not.toContain('min-w-0')

    const actions = container.querySelector('[data-selection-actions]') as HTMLElement
    expect(actions.className).toContain('min-w-0')
    expect(actions.className).toContain('flex-1')
    expect(actions.className).toContain('overflow-x-auto')
  })
})
