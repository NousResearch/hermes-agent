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
import { Loader2, MessageCircle } from '@/lib/icons'
import { notifyError } from '@/store/notifications'
import { $profiles, profileLabel } from '@/store/profile'
import { setSessionOwnerHint } from '@/store/session'
import { isSessionOwnerRoute } from '@/store/session-request-router'
import { isStartChatCallerWatched, readStartChatResult, takeLiveStartChat } from '@/store/start-chat'

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
  const callerId = useStore(useSessionView().$storedId)
  const profiles = useStore($profiles)
  const navigate = useNavigate()
  const outcome = useMemo(() => readStartChatResult(props.result), [props.result])
  const started = outcome?.status === 'started' ? outcome : null
  const args = parseMaybeObject(props.args)
  const message = typeof args.message === 'string' ? args.message.trim() : ''

  const title =
    started?.title || (typeof args.title === 'string' && args.title.trim()) || message.slice(0, TITLE_LIMIT) || null

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
      <ClarifyShell className="my-1.5 grid gap-0.5" data-slot="start-chat">
        <span className="font-medium">{copy.notStarted}</span>
        {outcome.reason ? <span className={CAPTION}>{outcome.reason}</span> : null}
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
