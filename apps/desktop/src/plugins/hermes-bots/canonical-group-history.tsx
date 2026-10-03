import { useI18n } from '@hermes/plugin-sdk'

import { type CanonicalGroupAttachment, CanonicalGroupAttachments } from './canonical-group-attachments'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalGroupBinding } from './canonical-groups'

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  kind: string
  payload: {
    text?: string; content?: string; attachments?: CanonicalGroupAttachment[]
    reason?: string; resource?: string; host_name?: string | null
    to_name?: string | null; from_name?: string | null; offline_since?: number | null
  }
  actor?: { kind?: string; id?: string; display_name?: string }
}

/** Bookkeeping kinds that carry nothing to show when empty. Unknown kinds always stay visible. */
const QUIET_KINDS = new Set(['turn.settled', 'room.activity', 'task.admitted', 'custody.configured', 'succession.state'])

type Labels = ReturnType<typeof useCanonicalGroupLabels>

function quiet(event: CanonicalGroupEvent) {
  return QUIET_KINDS.has(event.kind) && !event.payload.text && !event.payload.content && !event.payload.attachments?.length
}

/** Logged identities are authoritative; the Desktop's own unnamed actor uses the local label. */
function speaker(actor: CanonicalGroupEvent['actor'], you: string) {
  if (actor?.kind !== 'member' && actor?.kind !== 'user') {return undefined}

  return actor.display_name?.trim() || (actor.kind === 'user' && actor.id === 'desktop' ? you : actor.id)
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

/** The gateway words these notices in English; display labels let Desktop say them in the reader's language. */
function localizedNotice({ kind, payload }: CanonicalGroupEvent, labels: Labels, locale: string | undefined) {
  if (kind === 'authority.transition') {
    const target = label(payload.to_name), host = label(payload.from_name), time = offlineAt(payload.offline_since, locale)

    if (!target) {return Object.hasOwn(payload, 'to_name') ? labels.continuedOnUnnamed : undefined}

    return host && time ? fill(labels.continuedOnSince, { target, host, time }) : fill(labels.continuedOn, { target })
  }

  if (kind === 'turn.deferred' && payload.reason === 'waiting_for_host' && (payload.resource === 'bot' || payload.resource === 'file')) {
    const host = label(payload.host_name)

    if (!host) {return payload.resource === 'bot' ? labels.waitingForUnnamedHostBot : labels.waitingForUnnamedHostFile}

    return fill(payload.resource === 'bot' ? labels.waitingForHostBot : labels.waitingForHostFile, { host })
  }
}

export function CanonicalGroupHistory({ binding, events, disabled = false }: { binding: CanonicalGroupBinding; events: CanonicalGroupEvent[]; disabled?: boolean }) {
  const labels = useCanonicalGroupLabels()
  const { locale } = useI18n()

  return <>{events.filter(event => !quiet(event)).map(event => <div className="whitespace-pre-wrap py-2" key={event.event_id ?? event.seq}>
    {speaker(event.actor, labels.you) && <strong><bdi>{speaker(event.actor, labels.you)}</bdi>: </strong>}
    {localizedNotice(event, labels, locale) ?? (event.payload.text || event.payload.content || event.kind)}
    {!!event.payload.attachments?.length && <CanonicalGroupAttachments
      attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
      binding={binding} disabled={disabled || !event.event_id || event.room_id !== binding.roomId} readOnly />}
  </div>)}</>
}
