import { type ToolCallMessagePartProps } from '@assistant-ui/react'
import { type FC, useEffect, useState } from 'react'

import { AgentExchangeCard } from '@/components/assistant-ui/thread/agent-exchange-card'
import { AGENT_MESSAGE_RE, agentAvatarCache, resolveAgentAvatar } from '@/components/assistant-ui/thread/user-message'

// Sender-side inter-agent delivery: `hermes -p <agent> chat … -q "Message
// from 🤖 <sender>…"` run through the terminal tool IS the messaging
// pipeline (the Bot Mode / multi-profile convention shipped with #85855).
// Rendering it as a terminal transcript makes the sending bot's chat read
// like ops tooling; the user-facing truth is "Messaged X" and, when the
// quiet run returns the recipient's reply, "Message from X" — the same
// compact transcript-aligned event notices the receiving chat shows.
const DELIVERY_COMMAND_RE =
  /(?:^|[;&|]\s*|\bhermes\s+)-p\s+("?)([a-z0-9][a-z0-9_-]{0,63})\1\s+chat\b[\s\S]*?-q\s+["']Message from/iu

export function deliveryTargetFromCommand(command: string): null | string {
  const match = DELIVERY_COMMAND_RE.exec(command)

  return match ? match[2].toLowerCase() : null
}

/** Extract the recipient's reply text from the terminal result payload. */
export function replyTextFromResult(result: unknown): string {
  const container = (result ?? {}) as { content?: unknown; output?: unknown }
  let raw = ''

  if (typeof result === 'string') {
    raw = result
  } else if (typeof container.output === 'string') {
    raw = container.output
  } else if (Array.isArray(container.content)) {
    raw = container.content
      .map(entry => (typeof (entry as { text?: unknown })?.text === 'string' ? (entry as { text: string }).text : ''))
      .join('\n')
  }

  // Terminal results may be JSON-wrapped: {"output": "...", "exit_code": 0}
  if (raw.trimStart().startsWith('{')) {
    try {
      const parsed = JSON.parse(raw) as { output?: unknown }

      if (typeof parsed.output === 'string') {
        raw = parsed.output
      }
    } catch {
      /* not JSON — use as-is */
    }
  }

  // Drop session_id bookkeeping lines; what remains is the reply.
  return raw
    .split('\n')
    .filter(line => !/^session_id:\s/.test(line.trim()))
    .join('\n')
    .trim()
}

const AgentGlyph: FC<{ handle: string }> = ({ handle }) => {
  const [avatar, setAvatar] = useState<null | string>(() => agentAvatarCache.get(handle.toLowerCase()) ?? null)

  useEffect(() => {
    let live = true

    void resolveAgentAvatar(handle).then(url => {
      if (live && url) {
        setAvatar(url)
      }
    })

    return () => {
      live = false
    }
  }, [handle])

  return avatar ? (
    <img alt="" aria-hidden className="size-full object-cover" src={avatar} />
  ) : (
    <span aria-hidden className="text-[0.875rem] leading-none">
      🤖
    </span>
  )
}

/** "Messaged X" (+ "Message from X" once the reply lands) for a delivery
 *  command run via the terminal tool. Returns null when the command is not
 *  a delivery — caller falls through to the normal terminal row. */
export const AgentDeliveryNotice: FC<ToolCallMessagePartProps> = props => {
  const command = typeof props.args?.command === 'string' ? props.args.command : ''
  const target = deliveryTargetFromCommand(command)

  if (!target || props.isError) {
    return null
  }

  const pending = props.result === undefined
  const reply = pending ? '' : replyTextFromResult(props.result)
  // Strip a leading agent-message prefix if the recipient echoed one back.
  const replyBody = AGENT_MESSAGE_RE.exec(reply)?.[4] ?? reply

  return (
    <div className="flex w-full min-w-0 flex-col items-stretch gap-0.5">
      <AgentExchangeCard
        agent={target}
        avatar={<AgentGlyph handle={target} />}
        kind={pending ? 'sending' : 'sent'}
        slot="aui_agent-delivery-notice"
      />
      {!pending && replyBody && (
        <AgentExchangeCard
          agent={target}
          avatar={<AgentGlyph handle={target} />}
          body={<div className="whitespace-pre-wrap">{replyBody}</div>}
          bodyText={replyBody}
          kind="reply-from"
          slot="aui_agent-reply-notice"
        />
      )}
    </div>
  )
}
