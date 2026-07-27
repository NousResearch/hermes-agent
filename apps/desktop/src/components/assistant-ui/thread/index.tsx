import { type FC, useCallback, useMemo, useRef, useState } from 'react'

import { AssistantMessage } from '@/components/assistant-ui/thread/assistant-message'
import { ThreadMessageList } from '@/components/assistant-ui/thread/list'
import { BackgroundResumeNotice, CenteredThreadSpinner } from '@/components/assistant-ui/thread/status'
import { SystemMessage } from '@/components/assistant-ui/thread/system-message'
import { ThreadTimeline } from '@/components/assistant-ui/thread/timeline'
import { type RestoreMessageTarget } from '@/components/assistant-ui/thread/types'
import { UserEditComposer } from '@/components/assistant-ui/thread/user-edit-composer'
import { UserMessage } from '@/components/assistant-ui/thread/user-message'
import { Intro, type IntroProps } from '@/components/chat/intro'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'

type ThreadLoadingState = 'response' | 'session'

export const Thread: FC<{
  clampToComposer?: boolean
  cwd?: string | null
  gateway?: HermesGateway | null
  intro?: IntroProps
  loading?: ThreadLoadingState
  onBranchInNewChat?: (messageId: string) => void
  onCancel?: () => Promise<void> | void
  onDismissError?: (messageId: string) => void
  onRestoreToMessage?: (messageId: string, target?: RestoreMessageTarget) => Promise<void> | void
  sessionId?: string | null
  sessionKey?: string | null
}> = ({
  clampToComposer = false,
  cwd = null,
  gateway = null,
  intro,
  loading,
  onBranchInNewChat,
  onCancel,
  onDismissError,
  onRestoreToMessage,
  sessionId = null,
  sessionKey
}) => {
  const { t } = useI18n()
  const copy = t.assistant.thread

  const [restoreConfirmTarget, setRestoreConfirmTarget] = useState<
    (RestoreMessageTarget & { messageId: string }) | null
  >(null)

  const closeRestoreConfirm = useCallback(() => setRestoreConfirmTarget(null), [])

  const confirmRestore = useCallback(() => {
    if (!restoreConfirmTarget || !onRestoreToMessage) {
      throw new Error('Restore is unavailable for this message.')
    }

    const { messageId, text, userOrdinal } = restoreConfirmTarget

    closeRestoreConfirm()
    void Promise.resolve(onRestoreToMessage(messageId, { text, userOrdinal })).catch((error: unknown) => {
      notifyError(error, 'Restore failed')
    })
  }, [closeRestoreConfirm, onRestoreToMessage, restoreConfirmTarget])

  const requestRestoreConfirm = useCallback((messageId: string, target: RestoreMessageTarget) => {
    setRestoreConfirmTarget({ messageId, ...target })
  }, [])

  // The values in this map are component *types*: when their identity
  // changes, React unmounts and remounts every visible message — async
  // re-rendered parts (shiki code blocks) collapse and re-expand, so the
  // whole thread visibly jumps. Parents re-render on unrelated state
  // (e.g. the 15s status-snapshot poll in the desktop controller) and
  // can't be trusted to keep callback identities stable (see #38333), so
  // route the callbacks through a ref instead of listing them as memo
  // deps. Only their definedness stays a dep — it gates UI (the user
  // Stop button, the restore-confirm affordance). Assigned during render
  // (the useStoreSelector pattern) so the ref never lags a render.
  const callbacksRef = useRef({ onBranchInNewChat, onCancel, onDismissError, onRestoreToMessage })
  callbacksRef.current = { onBranchInNewChat, onCancel, onDismissError, onRestoreToMessage }

  const hasBranchInNewChat = Boolean(onBranchInNewChat)
  const hasCancel = Boolean(onCancel)
  const hasDismissError = Boolean(onDismissError)
  const hasRestoreToMessage = Boolean(onRestoreToMessage)

  const messageComponents = useMemo(
    () => ({
      AssistantMessage: () => (
        <AssistantMessage
          onBranchInNewChat={
            hasBranchInNewChat ? messageId => callbacksRef.current.onBranchInNewChat?.(messageId) : undefined
          }
          onDismissError={hasDismissError ? messageId => callbacksRef.current.onDismissError?.(messageId) : undefined}
        />
      ),
      SystemMessage,
      UserEditComposer: () => <UserEditComposer cwd={cwd} gateway={gateway} sessionId={sessionId} />,
      UserMessage: () => (
        <UserMessage
          onCancel={hasCancel ? () => callbacksRef.current.onCancel?.() : undefined}
          onRequestRestoreConfirm={hasRestoreToMessage ? requestRestoreConfirm : undefined}
        />
      )
    }),
    [
      cwd,
      gateway,
      hasBranchInNewChat,
      hasCancel,
      hasDismissError,
      hasRestoreToMessage,
      requestRestoreConfirm,
      sessionId
    ]
  )

  const emptyPlaceholder = intro ? (
    <div className="flex min-h-0 w-full flex-col items-center justify-center pt-[var(--composer-measured-height)]">
      <Intro {...intro} />
    </div>
  ) : undefined

  return (
    <div className="relative grid h-full min-h-0 max-w-full grid-rows-[minmax(0,1fr)] overflow-hidden bg-transparent contain-[layout_paint]">
      <ThreadMessageList
        clampToComposer={clampToComposer}
        components={messageComponents}
        emptyPlaceholder={emptyPlaceholder}
        loadingIndicator={<BackgroundResumeNotice />}
        sessionKey={sessionKey}
      />
      {loading === 'session' && <CenteredThreadSpinner />}
      <ThreadTimeline />
      <ConfirmDialog
        confirmLabel={copy.restoreConfirm}
        description={copy.restoreBody}
        destructive
        onClose={closeRestoreConfirm}
        onConfirm={confirmRestore}
        open={Boolean(restoreConfirmTarget)}
        title={copy.restoreTitle}
      />
    </div>
  )
}
