import type { ChatActions, SidebarActions } from './types'

/**
 * Surfaces receive one stable `actions` object whose fields are mutated by the
 * wiring controller each render. If a memoized surface passes `actions.foo`
 * directly, the child keeps the function from the surface's last render and can
 * submit/click against a stale session closure. These adapters keep a stable
 * wrapper but dereference the latest field at call time.
 *
 * OPTIONAL handlers must stay optional. Several children gate on a handler's
 * *presence*, not just call it — `onDismissError` renders the dismiss button
 * only when it exists (assistant-message.tsx), `onRestoreToMessage` gates the
 * restore-confirm flow (thread/index.tsx), and `onTranscribeAudio` gates voice
 * recording/conversation (use-voice-recorder, use-voice-conversation). An
 * unconditional arrow wrapper is always truthy, which would render a dead
 * dismiss button and let voice recording proceed with no transcription backend.
 * So wrap an optional field only when it is currently present, and re-read the
 * latest value inside the wrapper for the stale-closure fix.
 */
function latestOptional<A extends unknown[], R>(
  read: () => ((...args: A) => R) | undefined
): ((...args: A) => R) | undefined {
  // Presence is sampled from the object identity the surface currently holds.
  // The controller mutates fields in place rather than swapping a handler
  // between defined and undefined, so presence is stable for a given actions
  // object while the *closure* is what churns — which is exactly what the
  // indirection below re-reads.
  return read() ? (...args: A) => read()!(...args) : undefined
}

export function latestChatActions(actions: ChatActions): ChatActions {
  return {
    onAddContextRef: (...args) => actions.onAddContextRef(...args),
    onAddUrl: (...args) => actions.onAddUrl(...args),
    onAttachDroppedItems: (...args) => actions.onAttachDroppedItems(...args),
    onAttachImageBlob: (...args) => actions.onAttachImageBlob(...args),
    onBranchInNewChat: latestOptional(() => actions.onBranchInNewChat),
    onCancel: (...args) => actions.onCancel(...args),
    onDeleteSelectedSession: (...args) => actions.onDeleteSelectedSession(...args),
    onDismissError: latestOptional(() => actions.onDismissError),
    onEdit: (...args) => actions.onEdit(...args),
    onPasteClipboardImage: (...args) => actions.onPasteClipboardImage(...args),
    onPickFiles: (...args) => actions.onPickFiles(...args),
    onPickFolders: (...args) => actions.onPickFolders(...args),
    onPickImages: (...args) => actions.onPickImages(...args),
    onReload: (...args) => actions.onReload(...args),
    onRemoveAttachment: (...args) => actions.onRemoveAttachment(...args),
    onRestoreToMessage: latestOptional(() => actions.onRestoreToMessage),
    onRetryResume: (...args) => actions.onRetryResume(...args),
    onSteer: (...args) => actions.onSteer(...args),
    onSubmit: (...args) => actions.onSubmit(...args),
    onThreadMessagesChange: (...args) => actions.onThreadMessagesChange(...args),
    onToggleSelectedPin: (...args) => actions.onToggleSelectedPin(...args),
    onTranscribeAudio: latestOptional(() => actions.onTranscribeAudio)
  }
}

export function latestSidebarActions(actions: SidebarActions): SidebarActions {
  return {
    onArchiveSession: (...args) => actions.onArchiveSession(...args),
    onBranchSession: (...args) => actions.onBranchSession(...args),
    onDeleteSession: (...args) => actions.onDeleteSession(...args),
    onLoadMoreMessaging: latestOptional(() => actions.onLoadMoreMessaging),
    onLoadMoreSessions: (...args) => actions.onLoadMoreSessions(...args),
    onManageCronJob: (...args) => actions.onManageCronJob(...args),
    onNavigate: (...args) => actions.onNavigate(...args),
    onNewSessionInWorkspace: (...args) => actions.onNewSessionInWorkspace(...args),
    onNewSessionSplit: (...args) => actions.onNewSessionSplit(...args),
    onResumeSession: (...args) => actions.onResumeSession(...args),
    onTriggerCronJob: (...args) => actions.onTriggerCronJob(...args)
  }
}
