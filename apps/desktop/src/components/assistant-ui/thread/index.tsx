import { createContext, memo, useCallback, useContext, useMemo, useRef, useState } from 'react'

import { ChatEmptySlot } from '@/components/assistant-ui/chat-empty-slot'
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

interface ThreadEditContextValue {
  cwd: string | null
  gateway: HermesGateway | null
  sessionId: string | null
}

// Edit-composer context. The composer only exists while a message is being
// edited, and it mounts deep inside the memo'd ThreadMessageList, so the
// edit context can neither ride the component-map memo deps (that remints
// the component types on every session switch and remounts the outgoing
// transcript) nor sit in a render-time ref (a mounted composer never
// re-reads it when a same-session change leaves every list prop
// referentially equal). Context solves both: the component type stays
// stable, and a changed value propagates straight to the mounted consumer.
const ThreadEditContext = createContext<ThreadEditContextValue>({ cwd: null, gateway: null, sessionId: null })

interface ThreadProps {
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
}

// memo'd on purpose, and load-bearing for session-switch cost. ChatView
// re-renders on every route change (it reads `location`), and this subtree is
// the entire transcript — without a bail-out here the router's context update
// rebuilds every message of the OUTGOING thread before it is replaced. The
// props above are all stable across a plain re-render (see the component-map
// and loadingIndicator memos below), so the only thing that gets through is a
// genuine change.
export const Thread = memo(function Thread({
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
}: ThreadProps) {
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
  //
  // cwd / gateway / sessionId stay OUT of the memo deps for the same
  // reason: all three change on EVERY session switch, so listing them
  // re-minted these types mid-switch and remounted the entire OUTGOING
  // transcript — thousands of renders of a thread that was about to be
  // replaced, all of it before the resume RPC had even been sent. They
  // reach the edit composer through ThreadEditContext instead (see above).
  const callbacksRef = useRef({ onBranchInNewChat, onCancel, onDismissError, onRestoreToMessage })
  callbacksRef.current = { onBranchInNewChat, onCancel, onDismissError, onRestoreToMessage }

  // Only changes identity when one of the three values does, so Thread
  // re-renders for unrelated reasons never re-render the composer.
  const editContext = useMemo(() => ({ cwd, gateway, sessionId }), [cwd, gateway, sessionId])

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
      UserEditComposer: () => {
        const { cwd: editCwd, gateway: editGateway, sessionId: editSessionId } = useContext(ThreadEditContext)

        return <UserEditComposer cwd={editCwd} gateway={editGateway} sessionId={editSessionId} />
      },
      UserMessage: () => (
        <UserMessage
          onCancel={hasCancel ? () => callbacksRef.current.onCancel?.() : undefined}
          onRequestRestoreConfirm={hasRestoreToMessage ? requestRestoreConfirm : undefined}
        />
      )
    }),
    [hasBranchInNewChat, hasCancel, hasDismissError, hasRestoreToMessage, requestRestoreConfirm]
  )

  // Core's splash belongs to a fresh draft; a session that exists but has
  // nothing in it yet gets whichever plugin owns it. The slot often renders
  // nothing, which costs an empty container — harmless, since there is no
  // content to lay out until the first message swaps this branch out.
  const emptyBody = intro ? <Intro {...intro} /> : sessionId ? <ChatEmptySlot sessionId={sessionId} /> : null

  const emptyPlaceholder = emptyBody ? (
    <div className="flex min-h-0 w-full flex-col items-center justify-center pt-[var(--composer-measured-height)]">
      {emptyBody}
    </div>
  ) : undefined

  // Stable element identity, for the same reason the component map above is
  // memoized: this is a prop of the memo'd ThreadMessageList, so a fresh
  // element every render defeats the bail-out and drags the whole transcript
  // into the switch's render pass. It takes no props, so one element is
  // always correct.
  const loadingIndicator = useMemo(() => <BackgroundResumeNotice />, [])

  return (
    <ThreadEditContext.Provider value={editContext}>
      <div className="relative grid h-full min-h-0 max-w-full grid-rows-[minmax(0,1fr)] overflow-hidden bg-transparent contain-[layout_paint]">
        <ThreadMessageList
          clampToComposer={clampToComposer}
          components={messageComponents}
          emptyPlaceholder={emptyPlaceholder}
          loadingIndicator={loadingIndicator}
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
    </ThreadEditContext.Provider>
  )
})
