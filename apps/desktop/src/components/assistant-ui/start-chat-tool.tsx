'use client'

import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router'

import { useSessionView } from '@/app/chat/session-view'
import { openSessionFromPicker, type OpenSessionNavigate } from '@/app/open-session'
import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { CLARIFY_ICON_CLASS, ClarifyShell } from '@/components/assistant-ui/clarify/core/shell'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { parseMaybeObject } from '@/components/assistant-ui/tool/fallback-model/format'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { Loader2, MessageCircle } from '@/lib/icons'
import { notifyError } from '@/store/notifications'
import { $profiles, profileLabel } from '@/store/profile'
import { $sessions, sessionMatchesStoredId, setSessionOwnerHint } from '@/store/session'
import { isSessionOwnerRoute } from '@/store/session-request-router'
import {
  $startChatRetries,
  isStartChatCallerWatched,
  readStartChatResult,
  retryStartChat,
  takeLiveStartChat
} from '@/store/start-chat'

const TITLE_LIMIT = 40

const CAPTION = 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'

async function openStartedChat(
  chat: { profile: string; sessionId: string },
  callerId: null | string,
  navigate: OpenSessionNavigate,
  stillWanted: () => boolean = () => true
): Promise<void> {
  const owner = await resolveSessionOwner(callerId)

  if (isSessionOwnerRoute(owner)) {
    setSessionOwnerHint(chat.sessionId, { connectionId: owner.connectionId, profile: chat.profile })
  }

  if (stillWanted()) {
    openSessionFromPicker(chat.sessionId, navigate)
  }
}

export function StartChatTool(props: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.startChat
  const view = useSessionView()
  const callerId = useStore(view.$storedId)
  const callerRuntimeId = useStore(view.$runtimeId)
  const profiles = useStore($profiles)
  const sessions = useStore($sessions)
  const retry = useStore($startChatRetries)[props.toolCallId]
  const navigate = useNavigate()
  const recorded = useMemo(() => readStartChatResult(props.result), [props.result])
  const outcome = retry && retry !== 'pending' ? retry : recorded
  const started = outcome?.status === 'started' ? outcome : null
  const args = parseMaybeObject(props.args)
  const message = typeof args.message === 'string' ? args.message.trim() : ''
  const row = started ? sessions.find(session => sessionMatchesStoredId(session, started.sessionId)) : undefined

  const title = row
    ? sessionTitle(row)
    : started?.title || (typeof args.title === 'string' && args.title.trim()) || message.slice(0, TITLE_LIMIT) || null

  const retryChat = () => {
    if (!callerRuntimeId) {
      return
    }

    void retryStartChat(props.toolCallId, callerRuntimeId, {
      message: typeof args.message === 'string' ? args.message : '',
      profile: typeof args.profile === 'string' ? args.profile : null,
      title: typeof args.title === 'string' ? args.title : null
    })
      .then(next => (next?.status === 'started' ? openStartedChat(next, callerId, navigate) : undefined))
      .catch(error => notifyError(error, copy.notStarted))
  }

  useEffect(() => {
    if (!started || !takeLiveStartChat(props.toolCallId)) {
      return
    }

    const watching = () => Boolean(callerId) && isStartChatCallerWatched(callerId!)

    void openStartedChat(started, callerId, navigate, watching).catch(error => notifyError(error, copy.openFailed))
  }, [callerId, copy.openFailed, navigate, props.toolCallId, started])

  if (props.result !== undefined && !outcome) {
    return <ToolFallback {...props} />
  }

  if (outcome?.status === 'rejected') {
    return (
      <ClarifyShell className="my-1.5 flex items-center gap-2" data-slot="start-chat">
        <div className="grid min-w-0 flex-1 gap-0.5">
          <span className="font-medium">{copy.notStarted}</span>
          {outcome.reason ? <span className={CAPTION}>{outcome.reason}</span> : null}
        </div>
        <Button
          disabled={retry === 'pending' || !callerRuntimeId}
          onClick={retryChat}
          size="xs"
          type="button"
          variant="outline"
        >
          {retry === 'pending' ? (
            <Loader2 aria-hidden className="size-3.5 animate-spin motion-reduce:animate-none" />
          ) : null}
          {copy.retry}
        </Button>
      </ClarifyShell>
    )
  }

  if (!started) {
    return (
      <ClarifyShell className="my-1.5 flex items-center gap-2" data-slot="start-chat" role="status">
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary) motion-reduce:animate-none" />
        <span className="text-(--ui-text-tertiary)">{title ? copy.starting(title) : copy.startingUntitled}</span>
      </ClarifyShell>
    )
  }

  const profile = profiles.find(entry => entry.name === started.profile)

  return (
    <ClarifyShell className="my-1.5 flex items-center gap-2" data-slot="start-chat">
      <MessageCircle aria-hidden className={CLARIFY_ICON_CLASS} />
      <div className="grid min-w-0 flex-1">
        <span className="truncate font-medium">{title ?? copy.untitled}</span>
        {started.profile ? (
          <span className={CAPTION}>{copy.inProfile(profile ? profileLabel(profile) : started.profile)}</span>
        ) : null}
      </div>
      <Button
        onClick={() =>
          void openStartedChat(started, callerId, navigate).catch(error => notifyError(error, copy.openFailed))
        }
        size="xs"
        type="button"
        variant="outline"
      >
        {copy.open}
      </Button>
    </ClarifyShell>
  )
}
