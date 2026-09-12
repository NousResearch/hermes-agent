import { type CanonicalGroupAttachment, CanonicalGroupAttachments } from './canonical-group-attachments'
import type { CanonicalGroupBinding } from './canonical-groups'

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  kind: string
  payload: { text?: string; content?: string; attachments?: CanonicalGroupAttachment[] }
  actor?: { kind?: string; id?: string; member_id?: string; profile?: string; display_name?: string }
}

function speaker(event: CanonicalGroupEvent): string | undefined {
  const actor = event.actor

  // Event-owned metadata, never a same-named Bot from the foreground roster.
  return [actor?.display_name, actor?.id, actor?.member_id].find(value => typeof value === 'string' && value.trim())
}

export function CanonicalGroupHistory({ binding, events, disabled = false }: { binding: CanonicalGroupBinding; events: CanonicalGroupEvent[]; disabled?: boolean }) {
  return <>{events.map(event => <div className="whitespace-pre-wrap py-2" key={event.seq}>
    {speaker(event) && <strong>{speaker(event)}: </strong>}
    {event.payload.text || event.payload.content || event.kind}
    {!!event.payload.attachments?.length && <CanonicalGroupAttachments
      attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
      binding={binding} disabled={disabled || !event.event_id || event.room_id !== binding.roomId} readOnly />}
  </div>)}</>
}
