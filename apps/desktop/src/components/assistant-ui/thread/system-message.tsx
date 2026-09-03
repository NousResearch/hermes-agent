import { MessagePrimitive, useAuiState } from '@assistant-ui/react'
import { type FC } from 'react'

import { messageContentText } from '@/components/assistant-ui/thread/content'
import { MessageTimelineTimestamp } from '@/components/assistant-ui/thread/timeline-timestamp'
import { SCAFFOLD_LABEL_CLASS } from '@/components/chat/scaffold-row'
import { Codicon } from '@/components/ui/codicon'
import { ToolIcon } from '@/components/ui/tool-icon'
import { LinkifiedText } from '@/lib/external-link'
import { cn } from '@/lib/utils'

const SLASH_STATUS_RE = /^slash:(?<command>\/[^\n]+)\n(?<output>[\s\S]*)$/
const STEER_NOTE_RE = /^steer:(?<text>[\s\S]+)$/
const REVIEW_NOTE_RE = /^review:(?<label>[^:\n]+):?\s*(?<detail>[\s\S]*)$/

export const SystemMessage: FC = () => {
  const text = useAuiState(s => messageContentText(s.message.content))

  if (!text) {
    return null
  }

  // The self-improvement review saved something to memory/skills — the same
  // kind of event as a landed `memory` write, so it wears the same chrome:
  // brain glyph with the gold→purple glow, gradient label, purple detail,
  // left-aligned in the reading column like every other scaffold line.
  const reviewNote = text.match(REVIEW_NOTE_RE)

  if (reviewNote?.groups) {
    const detail = reviewNote.groups.detail.trim()

    return (
      <MessagePrimitive.Root
        className="flex w-full min-w-0 max-w-full items-start gap-1.5 self-start py-0.5"
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <span className="tool-memory-legendary-glyph flex h-(--conversation-line-height) w-3.5 shrink-0 items-center justify-center">
          <ToolIcon className="text-(--tool-memory-legendary-icon)" name="brain" size="0.875rem" />
        </span>
        <span className={cn(SCAFFOLD_LABEL_CLASS, 'tool-memory-legendary-title shrink-0 text-transparent')}>
          {reviewNote.groups.label.trim()}
        </span>
        {detail && (
          <span className={cn(SCAFFOLD_LABEL_CLASS, 'tool-memory-legendary-meta min-w-0 wrap-anywhere')}>{detail}</span>
        )}
      </MessagePrimitive.Root>
    )
  }

  const steerNote = text.match(STEER_NOTE_RE)

  if (steerNote?.groups) {
    return (
      <MessagePrimitive.Root
        className="flex max-w-[min(86%,44rem)] items-center gap-1.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60"
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <Codicon className="text-muted-foreground/55" name="compass" size="0.75rem" />
        <span className="text-muted-foreground/55">steered</span>
        <span className="text-muted-foreground/35">·</span>
        <span className="whitespace-pre-wrap">{steerNote.groups.text.trim()}</span> <MessageTimelineTimestamp />
      </MessagePrimitive.Root>
    )
  }

  const slashStatus = text.match(SLASH_STATUS_RE)

  if (slashStatus?.groups) {
    const output = slashStatus.groups.output.trim()
    // Single-line status (e.g. "model → x") reads best centered inline; padded
    // multiline output (catalogs, usage tables) needs left-aligned, wider room
    // or the column alignment breaks.
    const multiline = output.includes('\n')

    return (
      <MessagePrimitive.Root
        className={cn(
          'w-[60%] max-w-[44rem] self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60',
          multiline ? 'text-left' : 'text-center'
        )}
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <span className="font-mono text-muted-foreground/55">{slashStatus.groups.command}</span>
        {multiline ? (
          <LinkifiedText className="mt-0.5 block whitespace-pre-wrap" explicitOnly pretty={false} text={output} />
        ) : (
          <>
            <span className="mx-1.5 text-muted-foreground/35">·</span>
            <LinkifiedText className="whitespace-pre-wrap" explicitOnly pretty={false} text={output} />
          </>
        )}{' '}
        <MessageTimelineTimestamp className={cn(multiline ? 'mt-0.5 block' : 'ml-1.5')} />
      </MessagePrimitive.Root>
    )
  }

  const multiline = text.includes('\n')

  return (
    <MessagePrimitive.Root
      className={cn(
        'w-[60%] max-w-[44rem] self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/55',
        multiline ? 'text-left' : 'text-center'
      )}
      data-role="system"
      data-slot="aui_system-message-root"
    >
      <LinkifiedText className="whitespace-pre-wrap" explicitOnly pretty={false} text={text} />{' '}
      <MessageTimelineTimestamp className={cn(multiline ? 'mt-0.5 block' : 'ml-1.5')} />
    </MessagePrimitive.Root>
  )
}
