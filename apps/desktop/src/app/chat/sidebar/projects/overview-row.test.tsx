import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { I18nProvider, type Locale } from '@/i18n'

import { ProjectOverviewRow } from './overview-row'
import type { SidebarProjectTree } from './workspace-groups'

const modelState = vi.hoisted(() => ({ open: false, toggleOpen: vi.fn() }))

afterEach(() => {
  cleanup()
  modelState.open = false
  modelState.toggleOpen.mockReset()
})

vi.mock('./model', () => ({
  PROJECT_PREVIEW_COUNT: 3,
  latestProjectSessions: () => [],
  useWorkspaceNodeOpen: () => [modelState.open, modelState.toggleOpen]
}))

// ProjectMenu (the kebab) has its own dedicated test file — stub it here so
// this file only exercises overview-row's own Tip usage (the disclosure
// toggle) plus the WorkspaceAddButton wiring. ProjectContextMenu (the row's
// right-click wrapper) is stubbed as a pass-through so the row still renders.
vi.mock('./project-menu', () => ({
  ProjectContextMenu: ({ children }: { children: ReactNode }) => children,
  ProjectMenu: () => null
}))

const project = { id: 'p1', label: 'Test D' } as unknown as SidebarProjectTree

const tipTrigger = (el: HTMLElement) => el.closest('[data-slot="tooltip-trigger"]')

const renderLocalized = (node: ReactNode, locale: Locale = 'en') =>
  render(
    <I18nProvider configClient={null} initialLocale={locale}>
      {node}
    </I18nProvider>
  )

describe('ProjectOverviewRow', () => {
  it('wraps the "new session" add button in a Tip with the project-scoped label', () => {
    renderLocalized(<ProjectOverviewRow onNewSession={vi.fn()} project={project} />)

    const button = screen.getByRole('button', { name: 'New session in Test D' })
    expect(tipTrigger(button)).toBeTruthy()
  })

  it('wraps the disclosure toggle in a Tip when there are preview sessions', () => {
    renderLocalized(
      <ProjectOverviewRow
        previewSessions={[{ id: 's1' } as unknown as SessionInfo]}
        project={project}
        renderRows={() => null}
      />
    )

    // Collapsed by default, so the disclosure offers to show the sessions.
    const button = screen.getByRole('button', { name: 'Show Test D sessions' })
    expect(tipTrigger(button)).toBeTruthy()
  })

  it('uses the real Polish catalog with the target disclosure state passed by the row', () => {
    const props = {
      previewSessions: [{ id: 's1' } as unknown as SessionInfo],
      project,
      renderRows: () => null
    }

    const collapsed = renderLocalized(<ProjectOverviewRow {...props} />, 'pl')
    expect(screen.getByRole('button', { name: 'Pokaż sesje projektu „Test D”' })).toBeTruthy()
    collapsed.unmount()

    modelState.open = true
    renderLocalized(<ProjectOverviewRow {...props} />, 'pl')
    expect(screen.getByRole('button', { name: 'Ukryj sesje projektu „Test D”' })).toBeTruthy()
  })

  it('does not render the disclosure toggle when there is nothing to preview', () => {
    renderLocalized(<ProjectOverviewRow project={project} />)

    expect(screen.queryByRole('button', { name: 'Show Test D sessions' })).toBeNull()
  })

  it('offers the "new session" add button on Home, which starts one with no folder', () => {
    const home = {
      id: '__no_project__',
      isNoProject: true,
      label: 'Home',
      path: null
    } as unknown as SidebarProjectTree

    const onNewSession = vi.fn()

    renderLocalized(<ProjectOverviewRow onNewSession={onNewSession} project={home} />)
    fireEvent.click(screen.getByRole('button', { name: 'New session in Home' }))

    expect(onNewSession).toHaveBeenCalledWith(null)
  })

  it('tags the row with data-sessions-project so a skin can target one project', () => {
    const { container } = render(<ProjectOverviewRow project={project} />)

    expect(container.querySelector('[data-sessions-project="p1"]')).toBeTruthy()
  })
})
