import { describe, expect, it, vi } from 'vitest'

import { latestChatActions } from './latest-actions'
import type { ChatActions } from './types'

function makeActions(onSubmit: ChatActions['onSubmit']): ChatActions {
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
    onSubmit,
    onThreadMessagesChange: vi.fn(),
    onToggleSelectedPin: vi.fn(),
    onTranscribeAudio: vi.fn()
  }
}

describe('latestActions adapters', () => {
  it('dereferences the latest chat handler from a stable actions object', async () => {
    const staleSubmit = vi.fn(async () => false)
    const latestSubmit = vi.fn(async () => true)
    const actions = makeActions(staleSubmit)
    const adapted = latestChatActions(actions)

    actions.onSubmit = latestSubmit

    await expect(adapted.onSubmit('continue in selected session')).resolves.toBe(true)
    expect(staleSubmit).not.toHaveBeenCalled()
    expect(latestSubmit).toHaveBeenCalledWith('continue in selected session')
  })
})
