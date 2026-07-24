import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { SidebarWorkspaceGroup } from './workspace-group'

beforeAll(() => {
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  Element.prototype.scrollIntoView = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('SidebarWorkspaceGroup branch actions', () => {
  it('archives every session in the branch from its actions menu', async () => {
    const onArchiveSession = vi.fn()
    const sessions = [
      { id: 'session-a', started_at: 2 },
      { id: 'session-b', started_at: 1 }
    ]

    render(
      <I18nProvider configClient={null}>
        <SidebarWorkspaceGroup
          group={{
            id: '/repo/.worktrees/feature',
            label: 'feature/archive-all',
            mode: 'workspace',
            path: '/repo/.worktrees/feature',
            sessions: sessions as never
          }}
          onArchiveSession={onArchiveSession}
          renderRows={() => null}
        />
      </I18nProvider>
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Project actions' }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Archive all threads' }))

    await waitFor(() => {
      expect(onArchiveSession).toHaveBeenCalledTimes(2)
    })
    expect(onArchiveSession).toHaveBeenNthCalledWith(1, 'session-a')
    expect(onArchiveSession).toHaveBeenNthCalledWith(2, 'session-b')
  })
})
