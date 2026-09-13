import { type CanonicalGroupAttachment, CanonicalGroupAttachments } from './canonical-group-attachments'
import type { CanonicalGroupBinding } from './canonical-groups'

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  kind: string
  payload: { text?: string; content?: string; attachments?: CanonicalGroupAttachment[] }
  actor?: { member_id?: string }
}

export function CanonicalGroupHistory({ binding, events, disabled = false }: { binding: CanonicalGroupBinding; events: CanonicalGroupEvent[]; disabled?: boolean }) {
  return <>{events.map(event => <div className="whitespace-pre-wrap py-2" key={event.seq}>
    {event.actor?.member_id && <strong>{event.actor.member_id}: </strong>}
    {event.payload.text || event.payload.content || event.kind}
    {!!event.payload.attachments?.length && <CanonicalGroupAttachments
      attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
      binding={binding} disabled={disabled || !event.event_id || event.room_id !== binding.roomId} readOnly />}
  </div>)}</>
}
