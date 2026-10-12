'use client'

import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router'

import { useSessionView } from '@/app/chat/session-view'
import { openSessionFromPicker, type OpenSessionNavigate } from '@/app/open-session'
import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { parseMaybeObject } from '@/components/assistant-ui/tool/fallback-model/format'
import { useI18n } from '@/i18n'
import type { TimelinePartMetadata } from '@/lib/chat-messages/types'
import { sessionTitle } from '@/lib/chat-runtime'
import { useStoresSelector } from '@/lib/use-session-slice'
import { notifyError } from '@/store/notifications'
import { $profiles } from '@/store/profile'
import { $sessions, sessionMatchesStoredId, setSessionOwnerHint } from '@/store/session'
import { isSessionOwnerRoute } from '@/store/session-request-router'
import {
  $startChatRetries,
  isStartChatCallerWatched,
  retryStartChat,
  startChatOutcome,
  startChatRetry,
  startChatSuperseded,
  takeLiveStartChat
} from '@/store/start-chat'

import {
  requestedTitle,
  retryRequest,
  StartChatRejected,
  StartChatStarted,
  StartChatStarting
} from './start-chat-tool-parts'

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

export function StartChatTool(props: ToolCallMessagePartProps & Pick<TimelinePartMetadata, 'toolResultMetadata'>) {
  const { t } = useI18n()
  const copy = t.assistant.startChat
  const view = useSessionView()
  const callerId = useStore(view.$storedId)
  const callerRuntimeId = useStore(view.$runtimeId)
  const callerBusy = useStore(view.$busy)
  const profiles = useStore($profiles)
  const sessions = useStore($sessions)
  // Same fallback as the live-start mark in gateway-event/tools.ts, so both sides build the same key.
  const callerKey = callerId ?? callerRuntimeId ?? ''
  const retry = startChatRetry(useStore($startChatRetries), callerKey, props.toolCallId)
  const navigate = useNavigate()
  const { result, toolResultMetadata } = props

  const outcome = useMemo(
    () => startChatOutcome({ result, toolResultMetadata }, retry),
    [result, retry, toolResultMetadata]
  )

  const superseded = useStoresSelector(
    [view.$messages, $startChatRetries],
    () => outcome?.status === 'rejected' && startChatSuperseded(view.$messages.get(), callerKey, props.toolCallId)
  )

  const started = outcome?.status === 'started' ? outcome : null
  const args = parseMaybeObject(props.args)
  const row = started ? sessions.find(session => sessionMatchesStoredId(session, started.sessionId)) : undefined
  const title = row ? sessionTitle(row) : requestedTitle(started, args)

  const retryChat = () => {
    if (!callerRuntimeId) {
      return
    }

    void retryStartChat(callerKey, props.toolCallId, callerRuntimeId, retryRequest(args)).then(
      next => {
        if (next?.status === 'started') {
          void openStartedChat(next, callerId, navigate).catch(error => notifyError(error, copy.openFailed))
        }
      },
      error => notifyError(error, copy.notStarted)
    )
  }

  useEffect(() => {
    if (!started || !takeLiveStartChat(callerKey, props.toolCallId)) {
      return
    }

    const watching = () => Boolean(callerId) && isStartChatCallerWatched(callerId!)

    void openStartedChat(started, callerId, navigate, watching).catch(error => notifyError(error, copy.openFailed))
  }, [callerId, callerKey, copy.openFailed, navigate, props.toolCallId, started])

  if (props.result !== undefined && !outcome) {
    return <ToolFallback {...props} />
  }

  if (outcome?.status === 'rejected') {
    return (
      <StartChatRejected
        disabled={retry === 'pending' || callerBusy || !callerRuntimeId}
        onRetry={retryChat}
        pending={retry === 'pending'}
        reason={outcome.reason}
        showRetry={outcome.retryable && !superseded}
      />
    )
  }

  if (!started) {
    return <StartChatStarting title={title} />
  }

  return (
    <StartChatStarted
      onOpen={() =>
        void openStartedChat(started, callerId, navigate).catch(error => notifyError(error, copy.openFailed))
      }
      profile={started.profile}
      profiles={profiles}
      title={title}
    />
  )
}
