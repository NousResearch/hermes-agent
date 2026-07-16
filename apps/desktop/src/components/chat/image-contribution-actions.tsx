'use client'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useContributions } from '@/contrib/react/use-contributions'
import { cn } from '@/lib/utils'

export const CHAT_IMAGE_ACTIONS_AREA = 'chat.imageActions'

export interface ChatImageActionInput {
  src: string
  toolName?: string
}

export interface ChatImageActionContribution {
  codicon?: string
  label: string
  onSelect: (input: ChatImageActionInput) => void
  when?: (input: ChatImageActionInput) => boolean
}

function imageActionData(value: unknown): ChatImageActionContribution | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const candidate = value as Partial<ChatImageActionContribution>

  return typeof candidate.label === 'string' && typeof candidate.onSelect === 'function'
    ? (candidate as ChatImageActionContribution)
    : null
}

export function ChatImageActions({
  className,
  src,
  toolName
}: ChatImageActionInput & { className?: string }) {
  const contributions = useContributions(CHAT_IMAGE_ACTIONS_AREA)
  const input = { src, toolName }

  const actions = contributions
    .map(contribution => ({ contribution, data: imageActionData(contribution.data) }))
    .filter(
      (entry): entry is { contribution: (typeof contributions)[number]; data: ChatImageActionContribution } =>
        Boolean(entry.data && (!entry.data.when || entry.data.when(input)))
    )

  if (!src || actions.length === 0) {
    return null
  }

  return (
    <span className={cn('absolute left-2 top-2 z-10 flex gap-1', className)} data-slot="chat-image-actions">
      {actions.map(({ contribution, data }) => (
        <Button
          aria-label={data.label}
          key={contribution.id}
          onClick={event => {
            event.stopPropagation()
            data.onSelect(input)
          }}
          size="icon-sm"
          title={data.label}
          type="button"
          variant="secondary"
        >
          {data.codicon ? <Codicon name={data.codicon} /> : data.label.slice(0, 1)}
        </Button>
      ))}
    </span>
  )
}
