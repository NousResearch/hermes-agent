import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { $dismissedAutoProjectIds } from '@/store/layout'
import { $activeProjectId, $projectTree } from '@/store/projects'

import { ProjectSwitcherDialog } from './project-switcher'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      commandCenter: { projects: 'Projects' },
      projectSwitcher: {
        empty: 'No projects yet.',
        openFolder: 'Open folder…',
        searchPlaceholder: 'Search projects…',
        title: 'Switch project'
      }
    }
  })
}))

const project = (id: string, label: string, path: string, lastActive = 0): SidebarProjectTree => ({
  id,
  label,
  lastActive,
  path,
  repos: [],
  sessionCount: lastActive ? 1 : 0
})

beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
    configurable: true,
    value: vi.fn()
  })
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )
})

beforeEach(() => {
  $activeProjectId.set(null)
  $dismissedAutoProjectIds.set([])
  $projectTree.set([])
})

afterEach(cleanup)

describe('ProjectSwitcherDialog', () => {
  it('replaces rows when the active profile or connection publishes a new Projects tree', () => {
    $projectTree.set([project('local', 'Local project', '/local/repo')])

    render(<ProjectSwitcherDialog onOpenChange={vi.fn()} onOpenFolder={vi.fn()} onSelect={vi.fn()} open />)

    expect(screen.getByRole('option', { name: /Local project/ })).toBeTruthy()

    // `$projectTree` is the active gateway's cache. A profile or local/remote
    // re-home replaces it; the picker must follow that truth and retain no
    // window-global path list from the previous filesystem.
    act(() => $projectTree.set([project('remote', 'Remote project', '/srv/repo')]))

    expect(screen.queryByRole('option', { name: /Local project/ })).toBeNull()
    expect(screen.getByRole('option', { name: /Remote project/ })).toBeTruthy()
  })

  it('uses the shared overview ordering and selects a project id', () => {
    const onOpenChange = vi.fn()
    const onSelect = vi.fn()

    $activeProjectId.set('active')
    $projectTree.set([
      project('recent', 'Recently active', '/recent', 20),
      project('active', 'Pinned active', '/active', 10)
    ])

    render(<ProjectSwitcherDialog onOpenChange={onOpenChange} onOpenFolder={vi.fn()} onSelect={onSelect} open />)

    const rows = screen.getAllByRole('option')
    expect(rows[0]?.textContent).toContain('Pinned active')

    fireEvent.click(screen.getByRole('option', { name: /Pinned active/ }))

    expect(onSelect).toHaveBeenCalledWith('active')
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('clears the previous filter when the picker closes', () => {
    $projectTree.set([project('local', 'Local project', '/local/repo')])

    const props = { onOpenChange: vi.fn(), onOpenFolder: vi.fn(), onSelect: vi.fn() }
    const view = render(<ProjectSwitcherDialog {...props} open />)
    const input = screen.getByPlaceholderText('Search projects…') as HTMLInputElement

    fireEvent.change(input, { target: { value: 'open folder' } })
    expect(input.value).toBe('open folder')

    view.rerender(<ProjectSwitcherDialog {...props} open={false} />)
    view.rerender(<ProjectSwitcherDialog {...props} open />)

    expect((screen.getByPlaceholderText('Search projects…') as HTMLInputElement).value).toBe('')
    expect(screen.getByRole('option', { name: /Local project/ })).toBeTruthy()
  })
})
