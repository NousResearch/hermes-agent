import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $connection } from '@/store/session'

import { ProjectMenu } from './project-menu'
import { WorkspaceMenu } from './workspace-header'
import type { SidebarProjectTree } from './workspace-groups'

const project: SidebarProjectTree = {
  id: 'p_demo',
  label: 'Demo',
  path: '/repo/demo',
  repos: [],
  sessionCount: 0
}

function renderWithI18n(node: React.ReactNode) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      {node}
    </I18nProvider>
  )
}

describe('project and worktree reveal actions', () => {
  afterEach(() => {
    cleanup()
    $connection.set(null)
  })

  it('shows the project reveal action in local mode', async () => {
    $connection.set({ mode: 'local' } as never)

    renderWithI18n(<ProjectMenu isActive={false} project={project} />)

    fireEvent.pointerDown(screen.getAllByRole('button', { name: 'Project actions' })[0], { button: 0, ctrlKey: false })
    expect(await screen.findByText('Reveal in folder')).not.toBeNull()
  })

  it('shows the worktree reveal action in local mode', async () => {
    $connection.set({ mode: 'local' } as never)

    renderWithI18n(<WorkspaceMenu onRemove={() => undefined} path="/repo/demo/.worktrees/feature" />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Project actions' }), { button: 0, ctrlKey: false })
    expect(await screen.findByText('Reveal in folder')).not.toBeNull()
  })

  it('hides the project reveal action in remote mode', async () => {
    $connection.set({ mode: 'remote' } as never)

    renderWithI18n(<ProjectMenu isActive={false} project={project} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Project actions' }), { button: 0, ctrlKey: false })
    expect(await screen.findByText('Copy path')).not.toBeNull()
    expect(screen.queryByText('Reveal in folder')).toBeNull()
  })

  it('hides the worktree reveal action in remote mode', async () => {
    $connection.set({ mode: 'remote' } as never)

    renderWithI18n(<WorkspaceMenu onRemove={() => undefined} path="/repo/demo/.worktrees/feature" />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Project actions' }), { button: 0, ctrlKey: false })
    expect(await screen.findByText('Copy path')).not.toBeNull()
    expect(screen.queryByText('Reveal in folder')).toBeNull()
  })
})
