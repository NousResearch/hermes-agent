import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { SessionActionsMenu } from './session-actions-menu'

vi.mock('@/components/ui/copy-button', () => ({
  writeClipboardText: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/components/ui/dropdown-menu', () => ({
  DropdownMenu: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DropdownMenuContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DropdownMenuItem: ({
    children,
    disabled,
    onSelect
  }: {
    children: React.ReactNode
    disabled?: boolean
    onSelect?: (event: Event) => void
  }) => (
    <button
      disabled={disabled}
      onClick={() => onSelect?.(new Event('select'))}
      role="menuitem"
      type="button"
    >
      {children}
    </button>
  ),
  DropdownMenuTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('@/hermes', () => ({
  renameSession: vi.fn()
}))

vi.mock('@/lib/cloud-share', () => ({
  copyCloudChannelId: vi.fn(),
  deleteCloudChannel: vi.fn(),
  ensureCloudActionReady: vi.fn(),
  inviteCloudChannelMember: vi.fn(),
  loadCloudChannelMembers: vi.fn(),
  removeCloudChannelMember: vi.fn(),
  setCloudChannelMemberPermission: vi.fn(),
  shareSessionToCloud: vi.fn()
}))

vi.mock('@/lib/session-export', () => ({
  exportSession: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/session', () => ({
  setSessions: vi.fn()
}))

vi.mock('@/store/windows', () => ({
  canOpenSessionWindow: () => true,
  openSessionInNewWindow: vi.fn()
}))

const cloudShare = vi.mocked(await import('@/lib/cloud-share'))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('SessionActionsMenu ordering', () => {
  it('groups copy ids and keeps destructive deletes adjacent with standard Delete final', () => {
    render(
      <I18nProvider configClient={null}>
        <SessionActionsMenu
          onArchive={vi.fn()}
          onDelete={vi.fn()}
          onPin={vi.fn()}
          sessionId="session-123"
          title="Demo session"
        >
          <button type="button">Actions</button>
        </SessionActionsMenu>
      </I18nProvider>
    )

    const labels = screen
      .getAllByRole('menuitem')
      .map(item => item.textContent?.trim())
      .filter(Boolean)

    expect(labels.indexOf('Copy cloud ID')).toBe(labels.indexOf('Copy ID') + 1)
    expect(labels.slice(-3)).toEqual(['Archive', 'Delete cloud channel', 'Delete'])
  })

  it('checks cloud setup before opening invite-only cloud UI', async () => {
    cloudShare.ensureCloudActionReady.mockResolvedValue(null)

    render(
      <I18nProvider configClient={null}>
        <SessionActionsMenu sessionId="session-123" title="Demo session">
          <button type="button">Actions</button>
        </SessionActionsMenu>
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('menuitem', { name: /invite to cloud/i }))

    await waitFor(() => {
      expect(cloudShare.ensureCloudActionReady).toHaveBeenCalledWith('session-123', {
        requireShared: true,
        title: 'Invite to cloud'
      })
    })
    expect(screen.queryByPlaceholderText('name@example.com')).toBeNull()
  })
})
