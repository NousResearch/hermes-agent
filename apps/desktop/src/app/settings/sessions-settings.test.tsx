import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $sidebarSessionsOpenInNewTab } from '@/store/sidebar-open-preference'

const listAllProfileSessions = vi.fn()

vi.mock('@/hermes', () => ({
  deleteSession: vi.fn(),
  getHermesConfigRecord: vi.fn(),
  listAllProfileSessions: (...args: unknown[]) => listAllProfileSessions(...args),
  saveHermesConfig: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        sessions: {
          loading: 'Loading archived sessions…',
          archivedTitle: 'Archived sessions',
          archivedIntro: 'Archived chats are hidden from the sidebar.',
          emptyArchivedTitle: 'Nothing archived',
          emptyArchivedDesc: 'Archive a chat to hide it here.',
          unarchive: 'Unarchive',
          deletePermanently: 'Delete permanently',
          messages: (count: number) => `${count} messages`,
          restored: 'Restored',
          deleteConfirm: (title: string) => `Delete ${title}?`,
          autoArchiveTitle: 'Auto-archive stale chats',
          autoArchiveDesc: 'Automatically archive stale chats.',
          autoArchiveDaysLabel: 'Archive after',
          autoArchiveDaysUnit: 'days',
          autoArchiveFailed: 'Could not update auto-archive',
          defaultDirTitle: 'Default project directory',
          defaultDirDesc: 'Choose where new sessions start.',
          defaultDirUpdated: 'Default project directory updated',
          defaultsTo: (label: string) => `Defaults to ${label}.`,
          change: 'Change',
          choose: 'Choose',
          clear: 'Clear',
          notSet: 'Not set',
          failedLoad: 'Could not load archived sessions',
          unarchiveFailed: 'Unarchive failed',
          deleteFailed: 'Delete failed',
          updateDirFailed: 'Could not update default directory',
          clearDirFailed: 'Could not clear default directory',
          sidebarOpenInNewTabTitle: 'Open sidebar chats in new tabs',
          sidebarOpenInNewTabDesc: 'Ordinary sidebar clicks open/focus chats in tabs. Cmd/Ctrl-click still opens in a tab.'
        }
      }
    }
  })
}))

describe('SessionsSettings sidebar tab preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $sidebarSessionsOpenInNewTab.set(false)
    listAllProfileSessions.mockResolvedValue({ sessions: [] })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows a false-by-default toggle and updates the atom when toggled', async () => {
    const { SessionsSettings } = await import('./sessions-settings')
    render(
      <MemoryRouter>
        <SessionsSettings />
      </MemoryRouter>
    )

    await waitFor(() => expect(listAllProfileSessions).toHaveBeenCalledWith(200, 0, 'only'))

    const toggle = await screen.findByRole('switch', { name: 'Open sidebar chats in new tabs' })
    expect(toggle.getAttribute('data-state')).toBe('unchecked')

    fireEvent.click(toggle)

    expect($sidebarSessionsOpenInNewTab.get()).toBe(true)
  })
})
