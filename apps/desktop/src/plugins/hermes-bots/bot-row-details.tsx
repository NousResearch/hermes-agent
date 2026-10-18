import { cn } from '@hermes/plugin-sdk'

interface BotRowDetailsProps {
  /** The bot's circle; empty for the shared default circle. */
  circle: string
  fromBot: boolean
  /** The handle to show, or null when the row hides it. */
  handle: null | string
  preview: string
}

const dot = <span className="shrink-0 text-(--ui-text-quaternary)">·</span>

/** The row's second line: `@handle · circle · preview`, each part only when present. */
export function BotRowDetails({ circle, fromBot, handle, preview }: BotRowDetailsProps) {
  if (!handle && !circle && !preview && !fromBot) {
    return null
  }

  return (
    <div className="flex min-w-0 items-center gap-1.5 text-xs text-(--ui-text-tertiary)">
      {handle ? (
        <span className="shrink-0 font-mono text-[0.6875rem] text-(--ui-text-quaternary)">{`@${handle}`}</span>
      ) : null}
      {handle && circle ? dot : null}
      {circle ? (
        <span className="shrink-0 font-mono text-[0.6875rem] text-(--ui-text-quaternary)" data-slot="bot-circle">
          {circle}
        </span>
      ) : null}
      {(handle || circle) && preview ? dot : null}
      {preview ? <span className={cn('min-w-0 truncate', fromBot && 'italic')}>{preview}</span> : null}
    </div>
  )
}
