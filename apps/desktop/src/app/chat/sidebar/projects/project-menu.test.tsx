import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProjectMenu } from './project-menu'
import type { SidebarProjectTree } from './workspace-groups'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', confirm: 'Confirm', done: 'Done', loading: 'Loading…' },
      sidebar: {
        projects: {
          copyPath: 'Copy path',
          deleteConfirm: 'This cannot be undone.',
          menu: 'Project actions',
          menuAddFolder: 'Add folder',
          menuAppearance: 'Appearance',
          menuDelete: 'Delete',
          menuRename: 'Rename',
          menuSetActive: 'Set active',
          noColor: 'No color',
          removeFromSidebar: 'Remove from sidebar',
          reveal: 'Reveal in file manager'
        }
      }
    }
  })
}))

vi.mock('@/store/layout', () => ({
  $panesFlipped: {
    get: () => false,
    listen: () => () => {},
    subscribe: (fn: (v: boolean) => void) => {
      fn(false)

      return () => {}
    }
  },
  dismissAutoProject: vi.fn()
}))

vi.mock('@/store/projects', () => ({
  copyPath: vi.fn(),
  deleteProject: vi.fn(),
  openProjectAddFolder: vi.fn(),
  openProjectRename: vi.fn(),
  revealPath: vi.fn(),
  setActiveProject: vi.fn(),
  updateProject: vi.fn()
}))

const project = {
  color: null,
  icon: null,
  id: 'p1',
  isAuto: false,
  label: 'Test D',
  path: '/repo'
} as unknown as SidebarProjectTree

const tipTrigger = (el: HTMLElement) => el.closest('[data-slot="tooltip-trigger"]')

describe('ProjectMenu', () => {
  it('wraps the kebab trigger in a Tip', () => {
    render(<ProjectMenu isActive={false} project={project} />)

    const button = screen.getByRole('button', { name: 'Project actions' })
    expect(tipTrigger(button)).toBeTruthy()
  })

  // The 28-icon appearance grid (also wrapped in a per-icon Tip) sits behind
  // opening the dropdown menu, then the "Appearance" item, then the popover —
  // three chained Radix open-states that are exercised in the running app
  // (screenshot-verified) but are fragile to drive through jsdom/fireEvent
  // without real pointer-capture support. Not covered here; same code path
  // and pattern as the kebab tested above.
})
