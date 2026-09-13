import { MessagePrimitive, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { createContext, type ReactNode, useContext, useMemo } from 'react'

import { requestComposerInsert } from '@/app/chat/composer/focus'
import { useComposerScope, useComposerSurfaceId } from '@/app/chat/composer/scope'
import { messageAttachmentRefs, messageContentText } from '@/components/assistant-ui/thread/content'
import { useI18n } from '@/i18n'
import {
  $hiddenRecoveryEphemeral,
  $hiddenRecoveryRows,
  type RecoveryMessage,
  type RecoveryScope,
  type RecoveryStatus,
  recoveryStatuses,
  recoveryVisibilityKey,
  setRecoveryHidden
} from '@/lib/message-recovery'
import { isWatchWindow } from '@/store/windows'

const EMPTY_MESSAGES: readonly RecoveryMessage[] = []

const RecoveryContext = createContext<{ scope: RecoveryScope; statuses: ReadonlyMap<string, RecoveryStatus> } | null>(
  null
)

export function MessageRecoveryProvider({ children, scope }: { children: ReactNode; scope?: RecoveryScope }) {
  // No transcript subscription while streaming/loading. One reverse pass when
  // idle, never one full-history scan per user bubble on every token.
  const messages = useAuiState(s =>
    scope?.ready && !scope.pending && !s.thread.isRunning ? s.thread.messages : EMPTY_MESSAGES
  )

  const value = useMemo(() => (scope ? { scope, statuses: recoveryStatuses(messages) } : null), [messages, scope])

  return <RecoveryContext.Provider value={value}>{children}</RecoveryContext.Provider>
}

export function UserMessageRecovery({ children }: { children: ReactNode }) {
  const context = useContext(RecoveryContext)
  const message = useAuiState(s => s.message)
  const persisted = useStore($hiddenRecoveryRows)
  const ephemeral = useStore($hiddenRecoveryEphemeral)
  const { target } = useComposerScope()
  const surfaceId = useComposerSurfaceId()
  const { t } = useI18n()
  const copy = t.assistant.thread
  const identity = context ? recoveryVisibilityKey(context.scope, message) : null
  const hidden = identity && (identity.persistent ? persisted : ephemeral).includes(identity.key)
  const status = context?.statuses.get(message.id)

  if (!identity || isWatchWindow()) {
    return children
  }

  if (hidden) {
    return (
      <MessagePrimitive.Root
        className="flex items-center gap-2 py-1 text-xs text-muted-foreground"
        data-message-id={message.id}
        data-role="user"
      >
        <span>{copy.recoveryHidden}</span>
        <button
          className="rounded-md border border-border px-2 py-0.5 text-foreground hover:bg-muted disabled:opacity-50"
          onClick={() => setRecoveryHidden(identity, false)}
          type="button"
        >
          {copy.recoveryUndo}
        </button>
      </MessagePrimitive.Root>
    )
  }

  if (!status) {
    return children
  }

  const refs = messageAttachmentRefs(message.metadata?.custom?.attachmentRefs)
  const text = messageContentText(message.content)
  const prepared = text

  return (
    <>
      {children}
      <div
        className="mb-2 flex flex-wrap items-center justify-end gap-x-2 gap-y-1 text-xs text-muted-foreground"
        data-message-recovery={status}
      >
        <span>{status === 'unfinished' ? copy.recoveryUnfinished : copy.recoveryMissing}</span>
        <button
          className="rounded-md border border-border px-2 py-0.5 text-foreground hover:bg-muted disabled:opacity-50"
          disabled={!prepared || !surfaceId}
          onClick={() => requestComposerInsert(prepared, { mode: 'block', target, surfaceId: surfaceId! })}
          type="button"
        >
          {copy.recoveryPutBack}
        </button>
        <button
          className="rounded-md border border-border px-2 py-0.5 text-foreground hover:bg-muted disabled:opacity-50"
          onClick={() => setRecoveryHidden(identity, true)}
          title={copy.recoveryHideHint}
          type="button"
        >
          {copy.recoveryHide}
        </button>
        <span className="w-full text-right text-[0.6875rem]">
          {status === 'unfinished' ? copy.recoveryActivityHint : copy.recoveryHideHint}
          {refs.length ? ` ${copy.recoveryAttachmentHint}` : ''}
        </span>
      </div>
    </>
  )
}
