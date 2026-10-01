import { CopyButton, MessageTextContent, useI18n } from '@hermes/plugin-sdk'

import { type CanonicalGroupAttachment, CanonicalGroupAttachments } from './canonical-group-attachments'
import { CanonicalMemberFace, canonicalMemberName } from './canonical-group-identity'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalGroupBinding, CanonicalRoomMember } from './canonical-groups'

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  created_at?: number
  kind: string
  payload: { text?: string; content?: string; attachments?: CanonicalGroupAttachment[]; member_id?: string; error?: string; reason?: string }
  actor?: { kind?: string; id?: string; display_name?: string }
}

/** Bookkeeping kinds that carry nothing to show when empty. Unknown kinds always stay visible. */
const QUIET_KINDS = new Set(['turn.settled', 'room.activity'])

function quiet(event: CanonicalGroupEvent) {
  return QUIET_KINDS.has(event.kind) && !event.payload.text && !event.payload.content && !event.payload.attachments?.length
}

export function CanonicalGroupHistory({ binding, events, members = [], disabled = false }: {
  binding: CanonicalGroupBinding; events: CanonicalGroupEvent[]; members?: CanonicalRoomMember[]; disabled?: boolean
}) {
  const labels = useCanonicalGroupLabels()
  const { locale } = useI18n()

  return <>{events.filter(event => !quiet(event)).map(event => {
    const isBot = event.actor?.kind === 'member' || event.kind === 'message.member'
    const isHuman = event.actor?.kind === 'user' || event.kind === 'message.user'
    const member = members.find(candidate => candidate.member_id === (isBot ? event.actor?.id : event.payload.member_id))

    const name = isBot ? event.actor?.display_name?.trim() || canonicalMemberName(member, labels.unknownBot)
      : isHuman ? event.actor?.id === 'desktop' ? labels.you : event.actor?.display_name?.trim() || labels.unknownPerson : ''

    const activity: Record<string, string> = {
      'turn.failed': labels.activityFailed, 'turn.deferred': labels.activityDeferred,
      'turn.cancelled': labels.activityCancelled, 'member.unavailable': labels.activityUnavailable,
      'room.created': labels.activityCreated, 'room.disbanded': labels.activityEnded,
      'room.renamed': labels.activityRenamed, 'room.stop_requested': labels.stopped
    }

    const suppliedText = event.payload.text || event.payload.content
    const system = !suppliedText && !event.payload.attachments?.length

    const text = suppliedText || (system ? (activity[event.kind] || labels.activityUpdated)
      .replace('{name}', canonicalMemberName(member, labels.unknownBot)) : '')

    const timestamp = event.created_at && Number.isFinite(event.created_at)
      ? new Date(event.created_at < 1e12 ? event.created_at * 1000 : event.created_at) : null

    const at = timestamp && Number.isFinite(timestamp.getTime()) ? timestamp : null

    return <article className={`group flex min-w-0 items-start gap-3 py-3 ${isHuman ? 'rounded-lg bg-(--chrome-action-hover) px-3' : 'px-3'}`}
      key={event.event_id ?? event.seq}>
      {isBot && <div aria-hidden className="mt-0.5 shrink-0"><CanonicalMemberFace member={member} name={name} seed={event.actor?.id} /></div>}
      <div className="min-w-0 flex-1">
        {(name || at) && <div className="mb-1 flex min-w-0 items-center gap-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height)">
          {name && <span className="truncate font-medium text-(--ui-text-primary)"><bdi>{name}</bdi></span>}
          {at && <time className="shrink-0 text-(--ui-text-quaternary)" dateTime={at.toISOString()}>{new Intl.DateTimeFormat(locale, { hour: 'numeric', minute: '2-digit' }).format(at)}</time>}
          {!!text && <div className="ml-auto opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100"><CopyButton appearance="icon" buttonSize="icon-xs" text={text} /></div>}
        </div>}
        {!!text && (system ? <div className="select-text text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height)">
          <p className={event.kind === 'turn.failed' ? 'text-destructive' : 'text-(--ui-text-tertiary)'}>{text}</p>
          <details className="mt-1 text-(--ui-text-quaternary)"><summary className="cursor-pointer">{labels.setupDetails}</summary>
            <p className="mt-1 break-words font-mono text-[length:var(--conversation-tool-font-size)]">{event.kind}</p>
            {(event.payload.error || event.payload.reason) && <p className="mt-1 whitespace-pre-wrap break-words">{event.payload.error || event.payload.reason}</p>}
          </details>
        </div> : <div className="select-text break-words text-[length:var(--conversation-text-font-size)] leading-(--conversation-line-height)">
          <MessageTextContent media={false} previewOnly text={text} />
        </div>)}
        {!!event.payload.attachments?.length && <div className="mt-2"><CanonicalGroupAttachments
          attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
          binding={binding} disabled={disabled || !event.event_id || event.room_id !== binding.roomId} readOnly /></div>}
      </div>
    </article>
  })}</>
}
