'use client'

import { CLARIFY_ICON_CLASS, ClarifyShell } from '@/components/assistant-ui/clarify/core/shell'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Loader2, MessageCircle } from '@/lib/icons'
import { profileLabel } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

const TITLE_LIMIT = 40

const CAPTION = 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'

const stringArg = (value: unknown): null | string => (typeof value === 'string' ? value : null)

/** Card title before the session row exists: outcome title, then the requested title, then the message head. */
export function requestedTitle(
  started: null | { title?: null | string },
  args: Record<string, unknown>
): null | string {
  const message = stringArg(args.message)?.trim() ?? ''

  return started?.title || stringArg(args.title)?.trim() || message.slice(0, TITLE_LIMIT) || null
}

export function retryRequest(args: Record<string, unknown>) {
  return {
    message: stringArg(args.message) ?? '',
    profile: stringArg(args.profile),
    title: stringArg(args.title)
  }
}

interface StartChatRejectedProps {
  disabled: boolean
  onRetry: () => void
  pending: boolean
  reason?: null | string
  showRetry: boolean
}

export function StartChatRejected({ disabled, onRetry, pending, reason, showRetry }: StartChatRejectedProps) {
  const copy = useI18n().t.assistant.startChat

  return (
    <ClarifyShell className="my-1.5 flex items-center gap-2" data-slot="start-chat">
      <div className="grid min-w-0 flex-1 gap-0.5">
        <span className="font-medium">{copy.notStarted}</span>
        {reason ? <span className={CAPTION}>{reason}</span> : null}
      </div>
      {showRetry ? (
        <Button disabled={disabled} onClick={onRetry} size="xs" type="button" variant="outline">
          {pending ? <Loader2 aria-hidden className="size-3.5 animate-spin motion-reduce:animate-none" /> : null}
          {copy.retry}
        </Button>
      ) : null}
    </ClarifyShell>
  )
}

export function StartChatStarting({ title }: { title: null | string }) {
  const copy = useI18n().t.assistant.startChat

  return (
    <ClarifyShell className="my-1.5 flex items-center gap-2" data-slot="start-chat" role="status">
      <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary) motion-reduce:animate-none" />
      <span className="text-(--ui-text-tertiary)">{title ? copy.starting(title) : copy.startingUntitled}</span>
    </ClarifyShell>
  )
}

interface StartChatStartedProps {
  onOpen: () => void
  profile: string
  profiles: ProfileInfo[]
  title: null | string
}

export function StartChatStarted({ onOpen, profile, profiles, title }: StartChatStartedProps) {
  const copy = useI18n().t.assistant.startChat
  const entry = profiles.find(candidate => candidate.name === profile)

  return (
    <ClarifyShell className="my-1.5 flex items-center gap-2" data-slot="start-chat">
      <MessageCircle aria-hidden className={CLARIFY_ICON_CLASS} />
      <div className="grid min-w-0 flex-1">
        <span className="truncate font-medium">{title ?? copy.untitled}</span>
        {profile ? <span className={CAPTION}>{copy.inProfile(entry ? profileLabel(entry) : profile)}</span> : null}
      </div>
      <Button onClick={onOpen} size="xs" type="button" variant="outline">
        {copy.open}
      </Button>
    </ClarifyShell>
  )
}
