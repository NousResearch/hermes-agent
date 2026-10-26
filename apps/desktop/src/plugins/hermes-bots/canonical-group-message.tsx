import * as sdk from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import type { CanonicalGroupEvent } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'

const MessageTextContent = typeof sdk === 'undefined' ? undefined : sdk.MessageTextContent
const Streamdown = typeof sdk === 'undefined' ? undefined : sdk.Streamdown

function EventTime({ timestamp }: { timestamp?: number }) {
  if (typeof timestamp !== 'number') {return null}
  const at = new Date(timestamp * 1000)

  if (!Number.isFinite(at.getTime())) {return null}

  return <time className="text-[0.625rem] text-(--ui-text-quaternary)" dateTime={at.toISOString()}>{sdk.relativeTime(at.getTime())}</time>
}

function EventBody({ event, notice }: { event: CanonicalGroupEvent; notice?: string }) {
  if (notice) {
    const detail = event.payload.error || event.payload.reason || event.payload.reason_code

    return <p className="text-xs text-(--ui-text-secondary)" data-selectable-text="true">
      <strong>{notice}</strong>{detail && <>: {detail}</>}
    </p>
  }

  const text = event.payload.text || event.payload.content || ''

  return <div className="min-w-0 text-xs text-foreground [&_p]:mb-1 [&_p:last-child]:mb-0 [&_ul]:mb-1 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:mb-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_pre]:whitespace-pre-wrap [&_pre]:wrap-anywhere [&_[data-slot=code-card]>button]:opacity-100" data-selectable-text="true">
    {/* Room text may originate elsewhere: never resolve MEDIA: paths on this gateway. */}
    {MessageTextContent ? <MessageTextContent media={false} text={text} />
      : Streamdown ? <Streamdown>{text}</Streamdown> : text}
  </div>
}

export function CanonicalGroupMessage({ event, author, notice, children }: {
  event: CanonicalGroupEvent; author?: string; notice?: string; children?: ReactNode
}) {
  const labels = useCanonicalGroupLabels()
  const isUser = event.kind === 'message.user' || (event.kind === 'message' && event.actor?.kind === 'user')
  const text = event.payload.text || event.payload.content || ''
  const label = isUser && !event.actor?.display_name && (!author || author === 'desktop') ? labels.you : author

  return <div className={sdk.cn('grid grid-cols-[minmax(0,1fr)] gap-1 px-2 py-1.5',
    isUser ? 'ml-auto w-fit max-w-[85%] rounded-lg bg-(--chrome-action-hover)' : 'w-full')}
    data-message-role={isUser ? 'user' : notice ? 'system' : 'assistant'}>
    <div className="flex items-center gap-2">
      {!notice && <sdk.Codicon className="shrink-0 text-(--ui-text-tertiary)" name={isUser ? 'account' : 'hubot'} />}
      {label && <strong className={sdk.cn('min-w-0 wrap-anywhere text-[0.7rem] font-semibold', isUser ? 'text-foreground' : 'text-(--ui-accent)')}>{label}</strong>}
      <EventTime timestamp={event.created_at} />
      {!notice && text.trim() && <div className="ml-auto shrink-0"><sdk.CopyButton appearance="icon" buttonSize="icon" text={text} /></div>}
    </div>
    <EventBody event={event} notice={notice} />
    {children}
  </div>
}
