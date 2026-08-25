import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { ProjectOverviewRow } from './overview-row'
import type { SidebarProjectTree } from './workspace-groups'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        newSessionIn: (label: string) => `New session in ${label}`,
        projects: {
          enter: (label: string) => `Enter ${label}`,
          reorder: (label: string) => `Reorder ${label}`,
          toggle: (label: string, open: boolean) => `${open ? 'Show' : 'Hide'} ${label} sessions`
        }
      }
    }
  })
}))

const nodeOpen = vi.hoisted(() => ({ current: false }))

vi.mock('./model', () => ({
  PROJECT_PREVIEW_COUNT: 3,
  PROJECT_PREVIEW_LOADED: 10,
  latestProjectSessions: () => [],
  previewWindowMaxHeight: () => '86px',
  useWorkspaceNodeOpen: () => [nodeOpen.current, vi.fn()]
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

describe('ProjectOverviewRow', () => {
  it('wraps the "new session" add button in a Tip with the project-scoped label', () => {
    render(<ProjectOverviewRow onNewSession={vi.fn()} project={project} />)

    const button = screen.getByRole('button', { name: 'New session in Test D' })
    expect(tipTrigger(button)).toBeTruthy()
  })

  it('wraps the disclosure toggle in a Tip when there are preview sessions', () => {
    render(
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

  it('does not render the disclosure toggle when there is nothing to preview', () => {
    render(<ProjectOverviewRow project={project} />)

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

    render(<ProjectOverviewRow onNewSession={onNewSession} project={home} />)
    fireEvent.click(screen.getByRole('button', { name: 'New session in Home' }))

    expect(onNewSession).toHaveBeenCalledWith(null)
  })

  it('tags the row with data-sessions-project so a skin can target one project', () => {
    const { container } = render(<ProjectOverviewRow project={project} />)

    expect(container.querySelector('[data-sessions-project="p1"]')).toBeTruthy()
  })

  describe('preview window', () => {
    const sessions = (n: number) => Array.from({ length: n }, (_, i) => ({ id: `s${i}` }) as unknown as SessionInfo)

    // The nest is the only element that carries both rows and a maxHeight.
    const nest = (container: HTMLElement) => container.querySelector<HTMLElement>('[style*="max-height"]')

    beforeEach(() => {
      nodeOpen.current = true
    })

    afterEach(() => {
      nodeOpen.current = false
    })

    it('renders every loaded preview row, not just the three that fit', () => {
      const renderRows = vi.fn((_rows: SessionInfo[]) => null)

      render(<ProjectOverviewRow previewSessions={sessions(8)} project={project} renderRows={renderRows} />)

      expect(renderRows.mock.calls.at(-1)?.[0]).toHaveLength(8)
    })

    it('caps the preview at a fixed window and scrolls it once past three rows', () => {
      const { container } = render(
        <ProjectOverviewRow previewSessions={sessions(8)} project={project} renderRows={() => null} />
      )

      const window = nest(container)
      expect(window).toBeTruthy()
      expect(window?.style.maxHeight).toBe('86px')
      expect(window?.className).toContain('overflow-y-auto')
    })

    it('leaves a short preview unbounded — no window, no scroller', () => {
      const { container } = render(
        <ProjectOverviewRow previewSessions={sessions(3)} project={project} renderRows={() => null} />
      )

      expect(nest(container)).toBeNull()
    })

    it('never traps the wheel: the window must not contain its overscroll', () => {
      // The nest sits INSIDE the sidebar's own scroller — containing overscroll
      // here kills wheel chaining at the list's ends (#84964).
      const { container } = render(
        <ProjectOverviewRow previewSessions={sessions(8)} project={project} renderRows={() => null} />
      )

      expect(nest(container)?.className).not.toContain('overscroll-contain')
    })
  })
})
