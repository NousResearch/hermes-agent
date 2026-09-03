import { type FC, type ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

export type AgentExchangeKind = 'handoff' | 'reply-from' | 'reply-to' | 'sending' | 'sent'

const KIND_LABEL: Record<AgentExchangeKind, string> = {
  handoff: 'Handoff',
  'reply-from': 'Reply from',
  'reply-to': 'Reply to',
  sending: 'Sending to',
  sent: 'Sent to'
}

export function agentExchangePreview(text: string, maxLength = 120): string {
  const compact = text.replace(/\s+/g, ' ').trim()

  if (!compact) {
    return ''
  }

  const sentenceEnd = compact.search(/[.!?](?:\s|$)/)
  const firstSentence = sentenceEnd >= 0 ? compact.slice(0, sentenceEnd + 1) : compact

  if (firstSentence.length <= maxLength) {
    return firstSentence
  }

  return `${firstSentence.slice(0, Math.max(1, maxLength - 1)).trimEnd()}…`
}

export const AgentExchangeCard: FC<{
  agent: string
  avatar: ReactNode
  body?: ReactNode
  bodyText?: string
  className?: string
  kind: AgentExchangeKind
  replyProfile?: string
  slot: string
}> = ({ agent, avatar, body, bodyText = '', className, kind, replyProfile, slot }) => {
  const preview = agentExchangePreview(bodyText)
  const canOpenReply = Boolean(replyProfile && window.hermesDesktop?.quickEntry?.submit)

  const openReply = () => {
    if (!replyProfile || !window.hermesDesktop?.quickEntry?.submit) {
      return
    }

    window.hermesDesktop.quickEntry.submit({
      action: 'open-agent',
      profile: replyProfile,
      requestId: crypto.randomUUID()
    })
  }

  const identity = (
    <>
      <span className="mt-0.5 grid size-7 shrink-0 place-items-center overflow-hidden rounded-full border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)">
        {avatar}
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex min-w-0 items-baseline gap-1.5">
          <span className="shrink-0 text-[0.625rem] font-semibold uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
            {KIND_LABEL[kind]}
          </span>
          <span className="truncate text-[0.8125rem] font-semibold text-foreground/95">{agent}</span>
        </span>
        {preview && (
          <span className="mt-0.5 block truncate text-[0.75rem] leading-5 text-(--ui-text-secondary)">{preview}</span>
        )}
      </span>
    </>
  )

  return (
    <article
      className={cn(
        'w-full min-w-0 overflow-hidden rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-opaque-background) shadow-sm',
        className
      )}
      data-slot={slot}
    >
      {body ? (
        <details className="group/exchange">
          <summary
            aria-label={`${KIND_LABEL[kind]} ${agent}. Show full message`}
            className="flex cursor-pointer list-none items-start gap-2.5 px-3 py-2 text-left transition-colors hover:bg-(--ui-control-hover-background) [&::-webkit-details-marker]:hidden"
          >
            {identity}
            <Codicon
              className="mt-2 shrink-0 text-(--ui-text-tertiary) transition-transform duration-150 group-open/exchange:rotate-90"
              name="chevron-right"
              size="0.75rem"
            />
          </summary>
          <div className="border-t border-(--ui-stroke-tertiary) bg-(--ui-surface-background) px-3 py-2.5 text-left text-[0.8125rem] leading-5 text-foreground/90">
            {body}
          </div>
          {replyProfile && (
            <div className="flex justify-end border-t border-(--ui-stroke-tertiary) bg-(--ui-surface-background) px-3 py-1.5">
              <button
                aria-label={`Reply to ${agent} in their direct message`}
                className="inline-flex items-center gap-1.5 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-widget-surface-background) px-2.5 py-1 text-[0.75rem] font-medium text-foreground/90 shadow-sm transition-colors hover:border-(--ui-accent,#6e9fc5) hover:bg-(--ui-control-hover-background) disabled:cursor-not-allowed disabled:opacity-45"
                disabled={!canOpenReply}
                onClick={openReply}
                type="button"
              >
                <Codicon name="comment-discussion" size="0.75rem" />
                Reply in {agent} DM
              </button>
            </div>
          )}
        </details>
      ) : (
        <div className="flex items-start gap-2.5 px-3 py-2">
          {identity}
          {replyProfile && (
            <button
              aria-label={`Reply to ${agent} in their direct message`}
              className="shrink-0 rounded-md px-2 py-1 text-[0.75rem] font-medium text-(--ui-text-secondary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:cursor-not-allowed disabled:opacity-45"
              disabled={!canOpenReply}
              onClick={openReply}
              type="button"
            >
              Reply
            </button>
          )}
        </div>
      )}
    </article>
  )
}
