import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { SidebarWorkspaceGroup } from './workspace-group'
import type { SidebarSessionGroup } from './workspace-groups'

const modelState = vi.hoisted(() => ({ open: false, toggleOpen: vi.fn() }))

afterEach(() => {
  cleanup()
  modelState.open = false
  modelState.toggleOpen.mockReset()
})

vi.mock('./model', () => ({
  PROJECT_PREVIEW_COUNT: 3,
  SIDEBAR_GROUP_PAGE: 5,
  useWorkspaceNodeOpen: () => [modelState.open, modelState.toggleOpen]
}))

vi.mock('./workspace-header', () => ({
  WorkspaceAddButton: ({ label }: { label: string }) => <button type="button">{label}</button>,
  WorkspaceContextMenu: ({ children }: { children: ReactNode }) => children,
  WorkspaceHeader: () => null,
  WorkspaceMenu: () => null,
  WorkspaceShowMoreButton: () => null
}))

const group = {
  color: null,
  id: 'default',
  label: 'default',
  mode: 'profile',
  path: null,
  sessions: []
} as unknown as SidebarSessionGroup

const renderPolishGroup = () =>
  render(
    <I18nProvider configClient={null} initialLocale="pl">
      <SidebarWorkspaceGroup group={group} renderRows={() => null} />
    </I18nProvider>
  )

describe('SidebarWorkspaceGroup Polish disclosure label', () => {
  it('passes the target disclosure state to the real Polish catalog', () => {
    const collapsed = renderPolishGroup()
    expect(screen.getByRole('button', { name: 'Pokaż sesje projektu „default”' })).toBeTruthy()
    collapsed.unmount()

    modelState.open = true
    renderPolishGroup()
    expect(screen.getByRole('button', { name: 'Ukryj sesje projektu „default”' })).toBeTruthy()
  })
})
