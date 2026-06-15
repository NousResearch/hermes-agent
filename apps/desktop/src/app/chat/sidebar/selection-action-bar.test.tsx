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
  it('keeps the selected count on one line without clipping the action buttons', () => {
    $sidebarSelection.set({
      ids: ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'],
      section: 'sessions'
    })

    const { container } = render(
      <I18nProvider configClient={null}>
        <SelectionActionBar sessions={[]} />
      </I18nProvider>
    )

    const label = screen.getByLabelText('8 selected')
    expect(label.hasAttribute('data-selection-count-label')).toBe(true)
    expect(Array.from(label.children).map(child => child.textContent)).toEqual(['8', 'selected'])
    expect(label.className).toContain('whitespace-nowrap')
    expect(label.className).not.toContain('flex-wrap')
    expect(label.className).not.toContain('truncate')

    const countNumber = label.querySelector('[data-selection-count-number]') as HTMLElement
    expect(countNumber.className).toContain('shrink-0')

    const countText = label.querySelector('[data-selection-count-text]') as HTMLElement
    expect(countText.className).toContain('whitespace-nowrap')
    expect(countText.className).toContain('text-ellipsis')
    expect(countText.className).not.toContain('break-words')

    const count = container.querySelector('[data-selection-count]') as HTMLElement
    expect(count.className).toContain('min-w-0')

    const actions = container.querySelector('[data-selection-actions]') as HTMLElement
    expect(actions.className).toContain('shrink-0')
    expect(actions.className).not.toContain('min-w-0')
    expect(actions.className).not.toContain('flex-1')
    expect(actions.className).not.toContain('overflow-x-auto')

    expect(screen.getByLabelText('Prompt 8')).toBeTruthy()
    expect(screen.getByLabelText('Steer 8')).toBeTruthy()
    expect(screen.getByLabelText('Stop 8')).toBeTruthy()
  })
})
