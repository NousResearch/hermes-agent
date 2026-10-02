import { type CanonicalGroupAttachment, CanonicalGroupAttachments } from './canonical-group-attachments'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalGroupBinding } from './canonical-groups'

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  kind: string
  payload: { text?: string; content?: string; attachments?: CanonicalGroupAttachment[] }
  actor?: { kind?: string; id?: string; display_name?: string }
}

/** Bookkeeping kinds that carry nothing to show when empty. Unknown kinds always stay visible. */
const QUIET_KINDS = new Set(['turn.settled', 'room.activity'])

function quiet(event: CanonicalGroupEvent) {
  return QUIET_KINDS.has(event.kind) && !event.payload.text && !event.payload.content && !event.payload.attachments?.length
}

/** Logged identities are authoritative; the Desktop's own unnamed actor uses the local label. */
function speaker(actor: CanonicalGroupEvent['actor'], you: string) {
  if (actor?.kind !== 'member' && actor?.kind !== 'user') {return undefined}

  return actor.display_name?.trim() || (actor.kind === 'user' && actor.id === 'desktop' ? you : actor.id)
}

export function CanonicalGroupHistory({ binding, events, disabled = false }: { binding: CanonicalGroupBinding; events: CanonicalGroupEvent[]; disabled?: boolean }) {
  const labels = useCanonicalGroupLabels()

  return <>{events.filter(event => !quiet(event)).map(event => <div className="whitespace-pre-wrap py-2" key={event.event_id ?? event.seq}>
    {speaker(event.actor, labels.you) && <strong><bdi>{speaker(event.actor, labels.you)}</bdi>: </strong>}
    {event.payload.text || event.payload.content || event.kind}
    {!!event.payload.attachments?.length && <CanonicalGroupAttachments
      attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
      binding={binding} disabled={disabled || !event.event_id || event.room_id !== binding.roomId} readOnly />}
  </div>)}</>
}
