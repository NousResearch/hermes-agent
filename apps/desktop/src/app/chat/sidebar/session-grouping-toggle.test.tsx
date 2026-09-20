import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SessionGroupingToggle } from './session-grouping-toggle'

const { setSidebarGrouping } = vi.hoisted(() => ({ setSidebarGrouping: vi.fn() }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        groupAriaGrouped: 'Sessions are grouped by project',
        groupAriaUngrouped: 'Sessions are grouped by date',
        showProjects: 'Show Projects',
        showSessions: 'Show Sessions'
      }
    }
  })
}))

vi.mock('@/store/layout', async () => {
  const { atom } = await import('nanostores')
  const grouping = atom<'date' | 'project'>('date')
  setSidebarGrouping.mockImplementation((value: 'date' | 'project') => grouping.set(value))

  return { $sidebarGrouping: grouping, setSidebarGrouping }
})

vi.mock('@/store/projects', () => ({ exitProjectScope: vi.fn() }))

const layoutStore = await import('@/store/layout')

const grouping = layoutStore.$sidebarGrouping as typeof layoutStore.$sidebarGrouping & {
  set(value: 'date' | 'project'): void
}

afterEach(() => {
  cleanup()
  grouping.set('date')
  setSidebarGrouping.mockClear()
})

describe('SessionGroupingToggle', () => {
  it('stays visible, preserves composer focus, and toggles date and project grouping', async () => {
    render(
      <>
        <input aria-label="Composer" />
        <SessionGroupingToggle />
      </>
    )

    const composer = screen.getByRole('textbox', { name: 'Composer' })
    composer.focus()
    const showProjects = screen.getByRole('button', { name: 'Show Projects' })

    expect(showProjects.getAttribute('aria-description')).toBe('Sessions are grouped by date')
    expect(fireEvent.pointerDown(showProjects, { pointerType: 'mouse' })).toBe(false)
    fireEvent.click(showProjects)
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    expect(document.activeElement).toBe(composer)

    const showSessions = await screen.findByRole('button', { name: 'Show Sessions' })
    expect(showSessions.getAttribute('aria-description')).toBe('Sessions are grouped by project')
    fireEvent.click(showSessions)

    expect(await screen.findByRole('button', { name: 'Show Projects' })).toBeTruthy()
    expect(setSidebarGrouping.mock.calls).toEqual([['project'], ['date']])
  })
})
