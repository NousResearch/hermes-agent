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
  payload: {
    text?: string; content?: string; attachments?: CanonicalGroupAttachment[]; member_id?: string; error?: string; reason?: string
    resource?: string; host_name?: string | null
    to_name?: string | null; from_name?: string | null; offline_since?: number | null; successor_gateway_id?: string
  }
  actor?: { kind?: string; id?: string; display_name?: string }
}

/** Bookkeeping kinds that carry nothing to show when empty. Unknown kinds always stay visible. */
const QUIET_KINDS = new Set(['turn.settled', 'room.activity', 'task.admitted', 'custody.configured', 'succession.state'])

type Labels = ReturnType<typeof useCanonicalGroupLabels>

function quiet(event: CanonicalGroupEvent) {
  return QUIET_KINDS.has(event.kind) && !event.payload.text && !event.payload.content && !event.payload.attachments?.length
}

const label = (value: unknown) => typeof value === 'string' && value.trim() ? value.trim() : undefined

/** One pass, so a computer's display label is never read as another placeholder. */
const fill = (template: string, values: Record<string, string>) =>
  template.replace(/\{(\w+)\}/g, (token, key: string) => Object.hasOwn(values, key) ? values[key] : token)

function offlineAt(seconds: unknown, locale: string | undefined) {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds <= 0) {return undefined}
  const at = new Date(seconds * 1000)
  const today = at.toDateString() === new Date().toDateString()

  return new Intl.DateTimeFormat(locale, today ? { hour: 'numeric', minute: '2-digit' }
    : { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(at)
}

/** The gateway words these notices in English; display labels let Desktop say them in the reader's language.
 * A computer the event leaves unnamed may still be named by the room's latest status. */
function localizedNotice({ kind, payload }: CanonicalGroupEvent, labels: Labels, locale: string | undefined,
  computerName?: (installId: string) => string | undefined) {
  if (kind === 'authority.transition') {
    const named = typeof payload.successor_gateway_id === 'string' ? computerName?.(payload.successor_gateway_id) : undefined
    const target = label(payload.to_name) ?? named, host = label(payload.from_name), time = offlineAt(payload.offline_since, locale)

    if (!target) {return Object.hasOwn(payload, 'to_name') ? labels.continuedOnUnnamed : undefined}

    return host && time ? fill(labels.continuedOnSince, { target, host, time }) : fill(labels.continuedOn, { target })
  }

  if (kind === 'turn.deferred' && payload.reason === 'waiting_for_host' && (payload.resource === 'bot' || payload.resource === 'file')) {
    const host = label(payload.host_name)

    if (!host) {return payload.resource === 'bot' ? labels.waitingForUnnamedHostBot : labels.waitingForUnnamedHostFile}

    return fill(payload.resource === 'bot' ? labels.waitingForHostBot : labels.waitingForHostFile, { host })
  }
}

export function CanonicalGroupHistory({ binding, events, members = [], disabled = false, computerName }: {
  binding: CanonicalGroupBinding; events: CanonicalGroupEvent[]; members?: CanonicalRoomMember[]; disabled?: boolean
  computerName?: (installId: string) => string | undefined
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

    const notice = localizedNotice(event, labels, locale, computerName)
    const suppliedText = notice ? undefined : event.payload.text || event.payload.content
    const system = !!notice || !suppliedText && !event.payload.attachments?.length

    const text = notice || suppliedText || (system ? (activity[event.kind] || labels.activityUpdated)
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
