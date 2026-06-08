import { useStore } from '@nanostores/react'
import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'

import { sessionRoute } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import type { ChatMessage } from '@/lib/chat-messages'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { GitBranchIcon } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $forkOriginNotices, $messages, $sessions } from '@/store/session'

function ForkOriginNoticeRow({
  className,
  label,
  onOpenParent,
  openParentLabel
}: {
  className?: string
  label: string
  onOpenParent: () => void
  openParentLabel: string
}) {
  return (
    <div className={cn('flex items-center gap-3 py-1.5', className)}>
      <div aria-hidden className="h-px flex-1 bg-(--ui-stroke-tertiary)" />
      <Button
        aria-label={openParentLabel}
        className="pointer-events-auto h-auto gap-1.5 rounded-full px-2 py-0.5 text-[0.6875rem] text-(--ui-text-secondary) hover:text-foreground [-webkit-app-region:no-drag]"
        onClick={onOpenParent}
        size="xs"
        title={openParentLabel}
        type="button"
        variant="textStrong"
      >
        <GitBranchIcon className="size-3.5" />
        <span>{label}</span>
      </Button>
      <div aria-hidden className="h-px flex-1 bg-(--ui-stroke-tertiary)" />
    </div>
  )
}

function isBranchableMessage(message: ChatMessage): boolean {
  return !message.hidden && (message.role === 'assistant' || message.role === 'user')
}

export function SessionForkOriginNotice({
  className,
  messageId,
  storedSessionId
}: {
  className?: string
  messageId: string
  storedSessionId?: string | null
}) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const forkOriginNotices = useStore($forkOriginNotices)
  const messages = useStore($messages)
  const sessions = useStore($sessions)

  const currentNotice = storedSessionId ? forkOriginNotices[storedSessionId] : undefined

  const anchorMessageId = useMemo(() => {
    if (!currentNotice) {
      return null
    }

    const branchableMessages = messages.filter(isBranchableMessage)

    return branchableMessages[currentNotice.branchMessageOrdinal]?.id ?? null
  }, [currentNotice, messages])

  const parentSession = useMemo(() => {
    if (!currentNotice) {
      return null
    }

    return (
      sessions.find(
        session =>
          session.id === currentNotice.parentSessionId || session._lineage_root_id === currentNotice.parentSessionId
      ) ?? null
    )
  }, [currentNotice, sessions])

  if (!storedSessionId || !currentNotice || anchorMessageId !== messageId) {
    return null
  }

  const parentRouteSessionId = parentSession?.id ?? currentNotice.parentSessionId
  const parentLabel = parentSession ? sessionTitle(parentSession) : ''
  const openParentLabel = t.chat.openSourceConversation(parentLabel)

  return (
    <ForkOriginNoticeRow
      className={className}
      label={t.chat.forkOriginNotice}
      onOpenParent={() => {
        triggerHaptic('selection')
        navigate(sessionRoute(parentRouteSessionId))
      }}
      openParentLabel={openParentLabel}
    />
  )
}
