import { MessagePrimitive, useAuiState } from '@assistant-ui/react'
import { type FC } from 'react'

import { messageContentText } from '@/components/assistant-ui/thread/content'
import { Codicon } from '@/components/ui/codicon'
import { LinkifiedText } from '@/lib/external-link'
import { cn } from '@/lib/utils'

const SLASH_STATUS_RE = /^slash:(?<command>\/[^\n]+)\n(?<output>[\s\S]*)$/
const STEER_NOTE_RE = /^steer:(?<text>[\s\S]+)$/
// Model Router notes appended by the gateway-event handler:
//   "router:routed <tier>|<provider>:<model>"  — classifier decision
//   "router:fallback <from-label>|<to-label>"  — mid-turn fallback hop
const ROUTER_NOTE_RE = /^router:(?<kind>routed|fallback) (?<left>[^|]+)\|(?<right>[\s\S]+)$/

/** "lmstudio:qwen/qwen3-4b" → { provider: "lmstudio", model: "qwen/qwen3-4b" }. */
function splitSlotLabel(label: string): { provider: string; model: string } {
  const idx = label.indexOf(':')

  if (idx <= 0) {
    return { provider: '', model: label }
  }

  return { provider: label.slice(0, idx), model: label.slice(idx + 1) }
}

export const SystemMessage: FC = () => {
  const text = useAuiState(s => messageContentText(s.message.content))

  if (!text) {
    return null
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
        <span className="whitespace-pre-wrap">{steerNote.groups.text.trim()}</span>
      </MessagePrimitive.Root>
    )
  }

  const routerNote = text.match(ROUTER_NOTE_RE)

  if (routerNote?.groups) {
    const { kind, left, right } = routerNote.groups

    if (kind === 'routed') {
      const slot = splitSlotLabel(right.trim())

      return (
        <MessagePrimitive.Root
          className="flex max-w-[min(86%,44rem)] items-center gap-1.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60"
          data-role="system"
          data-slot="aui_system-message-root"
        >
          <Codicon className="text-muted-foreground/55" name="git-branch" size="0.75rem" />
          <span className="text-muted-foreground/55">routed</span>
          <span className="text-muted-foreground/35">·</span>
          <span>{left.trim()}</span>
          <span className="text-muted-foreground/35">·</span>
          <span className="truncate font-mono">{slot.model}</span>
          {slot.provider && <span className="shrink-0 text-muted-foreground/40">({slot.provider})</span>}
        </MessagePrimitive.Root>
      )
    }

    const from = splitSlotLabel(left.trim())
    const to = splitSlotLabel(right.trim())

    return (
      <MessagePrimitive.Root
        className="flex max-w-[min(86%,44rem)] items-center gap-1.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60"
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <Codicon className="text-muted-foreground/55" name="arrow-swap" size="0.75rem" />
        <span className="text-muted-foreground/55">rerouted</span>
        <span className="text-muted-foreground/35">·</span>
        <span className="truncate font-mono">{from.model}</span>
        <span className="text-muted-foreground/35">→</span>
        <span className="truncate font-mono">{to.model}</span>
        {to.provider && <span className="shrink-0 text-muted-foreground/40">({to.provider})</span>}
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
        )}
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
      <LinkifiedText className="whitespace-pre-wrap" explicitOnly pretty={false} text={text} />
    </MessagePrimitive.Root>
  )
}
