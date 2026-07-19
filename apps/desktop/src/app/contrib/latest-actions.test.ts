import { describe, expect, it, vi } from 'vitest'

import type { SidebarNavItem } from '../types'

import { latestChatActions, latestSidebarActions } from './latest-actions'
import type { ChatActions, SidebarActions } from './types'

function makeChatActions(): ChatActions {
  return {
    onAddContextRef: vi.fn(),
    onAddUrl: vi.fn(),
    onAttachDroppedItems: vi.fn(),
    onAttachImageBlob: vi.fn(),
    onBranchInNewChat: vi.fn(),
    onCancel: vi.fn(),
    onDeleteSelectedSession: vi.fn(),
    onDismissError: vi.fn(),
    onEdit: vi.fn(),
    onPasteClipboardImage: vi.fn(),
    onPickFiles: vi.fn(),
    onPickFolders: vi.fn(),
    onPickImages: vi.fn(),
    onReload: vi.fn(),
    onRemoveAttachment: vi.fn(),
    onRestoreToMessage: vi.fn(),
    onRetryResume: vi.fn(),
    onSteer: vi.fn(),
    onSubmit: vi.fn(),
    onThreadMessagesChange: vi.fn(),
    onToggleSelectedPin: vi.fn(),
    onTranscribeAudio: vi.fn()
  }
}

function makeSidebarActions(): SidebarActions {
  return {
    onArchiveSession: vi.fn(),
    onBranchSession: vi.fn(),
    onDeleteSession: vi.fn(),
    onLoadMoreMessaging: vi.fn(),
    onLoadMoreProfileSessions: vi.fn(),
    onLoadMoreSessions: vi.fn(),
    onManageCronJob: vi.fn(),
    onNavigate: vi.fn(),
    onNewSessionInWorkspace: vi.fn(),
    onNewSessionSplit: vi.fn(),
    onResumeSession: vi.fn(),
    onTriggerCronJob: vi.fn()
  }
}

describe('latestActions adapters', () => {
  it('dereferences the latest steer handler from a stable actions object', async () => {
    const staleSteer = vi.fn(async () => false)
    const latestSteer = vi.fn(async () => true)
    const actions = makeChatActions()
    actions.onSteer = staleSteer
    const adapted = latestChatActions(actions)

    actions.onSteer = latestSteer

    await expect(adapted.onSteer('continue in selected session')).resolves.toBe(true)
    expect(staleSteer).not.toHaveBeenCalled()
    expect(latestSteer).toHaveBeenCalledWith('continue in selected session')
  })

  it('dereferences the latest sidebar handler from a stable actions object', () => {
    const staleNavigate = vi.fn()
    const latestNavigate = vi.fn()
    const item = { id: 'settings', icon: vi.fn(), label: 'Settings', route: '/settings' } satisfies SidebarNavItem
    const actions = makeSidebarActions()
    actions.onNavigate = staleNavigate
    const adapted = latestSidebarActions(actions)

    actions.onNavigate = latestNavigate
    adapted.onNavigate(item)

    expect(staleNavigate).not.toHaveBeenCalled()
    expect(latestNavigate).toHaveBeenCalledWith(item)
  })
})
