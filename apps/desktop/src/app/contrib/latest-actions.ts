import type { ChatActions, SidebarActions } from './types'

/**
 * Surfaces receive one stable `actions` object whose fields are mutated by the
 * wiring controller each render. If a memoized surface passes `actions.foo`
 * directly, the child keeps the function from the surface's last render and can
 * submit/click against a stale session closure. These adapters keep a stable
 * wrapper but dereference the latest field at call time.
 */
export function latestChatActions(actions: ChatActions): ChatActions {
  return {
    onAddContextRef: (...args) => actions.onAddContextRef(...args),
    onAddUrl: (...args) => actions.onAddUrl(...args),
    onAttachDroppedItems: (...args) => actions.onAttachDroppedItems(...args),
    onAttachImageBlob: (...args) => actions.onAttachImageBlob(...args),
    onBranchInNewChat: (...args) => actions.onBranchInNewChat(...args),
    onCancel: (...args) => actions.onCancel(...args),
    onDeleteSelectedSession: (...args) => actions.onDeleteSelectedSession(...args),
    onDismissError: (...args) => actions.onDismissError?.(...args),
    onEdit: (...args) => actions.onEdit(...args),
    onPasteClipboardImage: (...args) => actions.onPasteClipboardImage(...args),
    onPickFiles: (...args) => actions.onPickFiles(...args),
    onPickFolders: (...args) => actions.onPickFolders(...args),
    onPickImages: (...args) => actions.onPickImages(...args),
    onReload: (...args) => actions.onReload(...args),
    onRemoveAttachment: (...args) => actions.onRemoveAttachment(...args),
    onRestoreToMessage: (...args) => actions.onRestoreToMessage?.(...args) ?? Promise.resolve(),
    onRetryResume: (...args) => actions.onRetryResume(...args),
    onSteer: (...args) => actions.onSteer(...args),
    onSubmit: (...args) => actions.onSubmit(...args),
    onThreadMessagesChange: (...args) => actions.onThreadMessagesChange(...args),
    onToggleSelectedPin: (...args) => actions.onToggleSelectedPin(...args),
    onTranscribeAudio: (...args) => actions.onTranscribeAudio?.(...args) ?? Promise.resolve('')
  }
}

export function latestSidebarActions(actions: SidebarActions): SidebarActions {
  return {
    onArchiveSession: (...args) => actions.onArchiveSession(...args),
    onBranchSession: (...args) => actions.onBranchSession(...args),
    onDeleteSession: (...args) => actions.onDeleteSession(...args),
    onLoadMoreMessaging: (...args) => actions.onLoadMoreMessaging?.(...args),
    onLoadMoreProfileSessions: (...args) => actions.onLoadMoreProfileSessions?.(...args),
    onLoadMoreSessions: (...args) => actions.onLoadMoreSessions(...args),
    onManageCronJob: (...args) => actions.onManageCronJob(...args),
    onNavigate: (...args) => actions.onNavigate(...args),
    onNewSessionInWorkspace: (...args) => actions.onNewSessionInWorkspace(...args),
    onNewSessionSplit: (...args) => actions.onNewSessionSplit(...args),
    onResumeSession: (...args) => actions.onResumeSession(...args),
    onTriggerCronJob: (...args) => actions.onTriggerCronJob(...args)
  }
}
