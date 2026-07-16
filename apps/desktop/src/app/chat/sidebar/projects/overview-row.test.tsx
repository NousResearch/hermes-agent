import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ProjectOverviewRow } from './overview-row'
import type { SidebarProjectTree } from './workspace-groups'

vi.mock('./project-menu', () => ({ ProjectMenu: () => null }))

const projectB: SidebarProjectTree = {
  id: 'p_project_b',
  label: 'Project B',
  path: '/work/project-b',
  repos: [],
  sessionCount: 0
}

function renderRow(overrides: Partial<React.ComponentProps<typeof ProjectOverviewRow>> = {}) {
  return render(
    <I18nProvider configClient={null}>
      <ProjectOverviewRow project={projectB} {...overrides} />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ProjectOverviewRow', () => {
  it('enters the clicked project before starting its new session', () => {
    const calls: string[] = []
    const onEnter = vi.fn((id: string) => calls.push(`enter:${id}`))
    const onNewSession = vi.fn((path: null | string) => calls.push(`new:${path}`))

    renderRow({ activeProjectId: 'p_project_a', onEnter, onNewSession })
    fireEvent.click(screen.getByRole('button', { name: 'New session in Project B' }))

    expect(calls).toEqual(['enter:p_project_b', 'new:/work/project-b'])
    expect(onEnter).toHaveBeenCalledWith(projectB.id)
    expect(onNewSession).toHaveBeenCalledWith(projectB.path)
  })

  it('keeps ordinary project entry separate from session creation', () => {
    const onEnter = vi.fn()
    const onNewSession = vi.fn()

    renderRow({ activeProjectId: 'p_project_a', onEnter, onNewSession })
    fireEvent.click(screen.getByRole('button', { name: 'Open Project B' }))

    expect(onEnter).toHaveBeenCalledWith(projectB.id)
    expect(onNewSession).not.toHaveBeenCalled()
  })
})
